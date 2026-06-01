#!/usr/bin/env python
# coding=utf-8
"""
Evaluate a multitask (classification + coordinate regression) model checkpoint.
- Loads the HF classification checkpoint (AutoModelForAudioClassification).
- Builds a small regression head with in_features == classifier.in_features.
- Loads regression weights from:
    1) --regression_head_path (if provided), else
    2) <checkpoint>/model.safetensors (extract keys starting with 'regression_head.'), else
    3) parent run directory's regression_head.pt (if present).
- Uses a classifier pre-hook to capture the pooled features (same representation as training).
- Reports classification accuracy + regression metrics (MSE/MAE per coord + Haversine distance).
"""

import os
import json
import argparse
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, Audio
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from safetensors.torch import load_file as safe_load

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Normalization constants (kept in sync with training) ---
LAT_DIVISOR = 90.0
LON_DIVISOR = 180.0

def great_circle_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> np.ndarray:
    """Haversine distance in kilometers."""
    R = 6371.0088
    lat1 = np.radians(lat1_deg); lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg); lon2 = np.radians(lon2_deg)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_classifier_in_features(model: nn.Module) -> int:
    """Return the dimension of the tensor that feeds the classifier."""
    clf = getattr(model, "classifier", None)
    if isinstance(clf, nn.Linear):
        return clf.in_features
    # Some heads wrap a linear as .out_proj
    if hasattr(clf, "out_proj") and isinstance(clf.out_proj, nn.Linear):
        return clf.out_proj.in_features
    # Fallbacks (rare)
    for key in ["hidden_size", "d_model", "proj_dim"]:
        if hasattr(model.config, key):
            v = getattr(model.config, key)
            if isinstance(v, int) and v > 0:
                return v
    proj = getattr(model, "projector", None)
    if isinstance(proj, nn.Linear):
        return proj.in_features
    raise ValueError("Could not infer classifier input dimension.")

def load_dataset(path: str, label2id: Dict[str, int], label_column: str,
                 audio_column: str, samp_rate: int) -> Dataset:
    df = pd.read_pickle(path)
    df.reset_index(drop=True, inplace=True)
    df["class_label"] = df[label_column].apply(lambda x: label2id[x])
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column(audio_column, Audio(sampling_rate=samp_rate))
    ds = ds.rename_column(audio_column, "audio")
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/idatro/dialect_project/repo/did_prosody_whisper-main/hid64-64_a1.0_fr_model_output/nordia_fgdid_h64-64_a1.0_t1_a1.0/checkpoint-1282", help="Path to checkpoint dir (e.g., .../checkpoint-480) or run root")
    parser.add_argument("--test_dataset", type=str, default="/home/idatro/dialect_project/ndc_folds_loc/fold_pkls/fold1.pkl")
    parser.add_argument("--save_path", type=str, default="/home/idatro/dialect_project/repo/did_prosody_whisper-main/hid64-64_a1.0_fr_model_output/nordia_fgdid_h64-64_a1.0_t1_a1.0/eval_ep4")
    parser.add_argument("--label_column_name", type=str, default="fg_dialect_region")
    parser.add_argument("--audio_column_name", type=str, default="full_audio_file_path")
    parser.add_argument("--offset_start_column", type=str, default="offset_start_ms")
    parser.add_argument("--offset_end_column", type=str, default="offset_end_ms")
    parser.add_argument("--chunk_seconds", type=float, default=30.0)
    parser.add_argument("--latitude_column", type=str, default="latitude")
    parser.add_argument("--longitude_column", type=str, default="longitude")
    parser.add_argument("--regression_head_path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model from:", args.model_path)
    model = AutoModelForAudioClassification.from_pretrained(args.model_path).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)

    # --- Load multitask meta (from checkpoint dir or its parent run dir) ---
    meta_path = os.path.join(args.model_path, "multitask_meta.json")
    if not os.path.isfile(meta_path):
        parent = os.path.dirname(args.model_path.rstrip("/"))
        meta_path = os.path.join(parent, "multitask_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        global LAT_DIVISOR, LON_DIVISOR
        LAT_DIVISOR = float(meta.get("latitude_divisor", LAT_DIVISOR))
        LON_DIVISOR = float(meta.get("longitude_divisor", LON_DIVISOR))
        args.latitude_column = meta.get("latitude_column", args.latitude_column)
        args.longitude_column = meta.get("longitude_column", args.longitude_column)

    # --- Build regression head with classifier input dim ---
    in_dim = get_classifier_in_features(model)
    dropout_p = (
        getattr(model.config, "classifier_dropout_prob", None)
        or getattr(model.config, "hidden_dropout_prob", None)
        or 0.1
    )
    regression_head = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, 64), # HIDDEN LAYER = 64
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(64, 64),  # second hidden layer
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(64, 2),
    ).to(device)

    # --- Load regression head weights ---
    loaded_head = False
    # 1) explicit sidecar path or sidecar inside model_path
    sidecar_path = args.regression_head_path or os.path.join(args.model_path, "regression_head.pt")
    if os.path.isfile(sidecar_path):
        regression_head.load_state_dict(torch.load(sidecar_path, map_location=device))
        loaded_head = True
    else:
        # 2) extract from safe tensors inside this checkpoint dir
        st_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.isfile(st_path):
            sd = safe_load(st_path, device="cpu")
            sub = {k.split("regression_head.", 1)[1]: v
                   for k, v in sd.items() if k.startswith("regression_head.")}
            if sub:
                regression_head.load_state_dict(sub, strict=True)
                loaded_head = True
    # 3) final fallback: sidecar placed in the parent run root
    if not loaded_head:
        parent = os.path.dirname(args.model_path.rstrip("/"))
        sidecar_parent = os.path.join(parent, "regression_head.pt")
        if os.path.isfile(sidecar_parent):
            regression_head.load_state_dict(torch.load(sidecar_parent, map_location=device))
            loaded_head = True

    if not loaded_head:
        # Helpful debug: list candidate keys
        st_path = os.path.join(args.model_path, "model.safetensors")
        hint = ""
        if os.path.isfile(st_path):
            sd = safe_load(st_path, device="cpu")
            candidates = [k for k in sd.keys() if "regression" in k]
            hint = f"\nFound keys containing 'regression': {candidates[:10]}"
        raise FileNotFoundError(
            "Could not locate regression head weights.\n"
            f"  tried sidecar: {sidecar_path}\n"
            f"  tried safetensors: {st_path}\n"
            f"  tried parent sidecar: {os.path.join(os.path.dirname(args.model_path.rstrip('/')),'regression_head.pt')}"
            + hint
        )
    regression_head.eval()

    # --- Register classifier pre-hook to capture pooled features (B, D_in) ---
    hf_model = model
    hf_model._pooled_cache = None
    def _store_classifier_input(module, inputs):
        hf_model._pooled_cache = inputs[0]  # (B, in_dim)
    if not hasattr(hf_model, "classifier"):
        raise RuntimeError("Expected `model.classifier` to exist on the audio classification model.")
    hf_model._clf_hook = hf_model.classifier.register_forward_pre_hook(_store_classifier_input)

    # --- Label maps from checkpoint config ---
    label2id = {k: int(v) for k, v in model.config.label2id.items()}
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels_ordered: List[str] = [id2label[i] for i in sorted(id2label.keys())]

    # --- Prepare dataset ---
    print("Loading test dataset from:", args.test_dataset)
    test_ds = load_dataset(
        args.test_dataset,
        label2id,
        label_column=args.label_column_name,
        audio_column=args.audio_column_name,
        samp_rate=feature_extractor.sampling_rate,
    )

    # --- Prediction loop (single-sample for simplicity; can be batched) ---
    y_true, y_pred = [], []
    lat_true_deg, lon_true_deg, lat_pred_deg, lon_pred_deg = [], [], [], []

    def slice_and_prepare(item: Dict[str, Any]):
        wav = item["audio"]["array"]; sr = item["audio"]["sampling_rate"]
        s = item.get(args.offset_start_column, 0.0)
        e = item.get(args.offset_end_column, None)
        s_sec = float(s) * (1e-3 if str(args.offset_start_column).endswith("_ms") else 1.0)
        if e is None:
            e_sec = min(s_sec + args.chunk_seconds, len(wav) / sr)
        else:
            e_sec = float(e) * (1e-3 if str(args.offset_end_column).endswith("_ms") else 1.0)
        st = max(0, int(round(s_sec * sr))); en = min(len(wav), int(round(e_sec * sr)))
        seg = wav[st:en]; 
        if seg.size == 0:
            seg = wav[: int(sr * args.chunk_seconds)]
        max_len = int(round(feature_extractor.sampling_rate * args.chunk_seconds))
        return feature_extractor(
            seg,
            sampling_rate=sr,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=False,   # not needed with pre-hook path
        )

    model.eval()
    for item in tqdm(test_ds, desc="Predicting"):
        inputs = slice_and_prepare(item)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)  # no output_hidden_states
            logits = outputs["logits"]
            pred_id = torch.argmax(logits, dim=-1).item()

            # pooled features captured by pre-hook
            pooled = getattr(hf_model, "_pooled_cache", None)
            if pooled is None:
                raise RuntimeError("Classifier pre-hook did not capture pooled features.")
            # shape guard
            lin = regression_head[1] if isinstance(regression_head, nn.Sequential) else regression_head
            if pooled.size(-1) != lin.in_features:
                raise RuntimeError(f"Pooled dim {pooled.size(-1)} != regression head in_features {lin.in_features}")

            coords_norm = regression_head(pooled).squeeze(0).detach().cpu().numpy()
            lat_pred_deg.append(float(coords_norm[0] * LAT_DIVISOR))
            lon_pred_deg.append(float(coords_norm[1] * LON_DIVISOR))

            y_pred.append(pred_id)
            y_true.append(item["class_label"])
            lat_true_deg.append(float(item[args.latitude_column]))
            lon_true_deg.append(float(item[args.longitude_column]))

    # --- Metrics ---
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)

    lat_true = np.array(lat_true_deg); lon_true = np.array(lon_true_deg)
    lat_pred = np.array(lat_pred_deg); lon_pred = np.array(lon_pred_deg)

    mse_lat = mean_squared_error(lat_true, lat_pred)
    mse_lon = mean_squared_error(lon_true, lon_pred)
    mae_lat = mean_absolute_error(lat_true, lat_pred)
    mae_lon = mean_absolute_error(lon_true, lon_pred)
    hav_km = great_circle_km(lat_true, lon_true, lat_pred, lon_pred)
    mean_hav_km = float(np.mean(hav_km)); median_hav_km = float(np.median(hav_km))

    print(f"MSE (lat, lon): ({mse_lat:.4f}, {mse_lon:.4f})")
    print(f"MAE (lat, lon): ({mae_lat:.4f}, {mae_lon:.4f})")
    print(f"Haversine distance: mean={mean_hav_km:.2f} km, median={median_hav_km:.2f} km")

    # --- Persist predictions + metrics ---
    df = test_ds.to_pandas()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    df["predicted_label"] = [id2label[i] for i in y_pred]
    df["pred_lat_deg"] = lat_pred_deg; df["pred_lon_deg"] = lon_pred_deg

    out_base = os.path.join(args.save_path, os.path.basename(args.model_path.rstrip("/")))
    os.makedirs(out_base, exist_ok=True)
    df.to_pickle(os.path.join(out_base, "predictions.pkl"))

    with open(os.path.join(out_base, "metrics.txt"), "w") as f:
        f.write(
            f"Accuracy: {acc}\n"
            f"MSE_lat: {mse_lat}\nMSE_lon: {mse_lon}\n"
            f"MAE_lat: {mae_lat}\nMAE_lon: {mae_lon}\n"
            f"Haversine_mean_km: {mean_hav_km}\nHaversine_median_km: {median_hav_km}\n"
        )

    # --- Confusion matrix ---
    try:
        labels_ordered = [id2label[i] for i in sorted(id2label.keys())]
        cm = confusion_matrix([id2label[i] for i in y_true], [id2label[i] for i in y_pred], labels=labels_ordered)
    except ValueError:
        cm = confusion_matrix([id2label[i] for i in y_true], [id2label[i] for i in y_pred])
        labels_ordered = None
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-8)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, vmin=0, vmax=1, linecolor="white", annot=True, fmt=".2f",
                linewidths=0.5, cmap="Blues",
                xticklabels=labels_ordered, yticklabels=labels_ordered)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "conf_matrix.png"))

    # --- Haversine histogram ---
    plt.figure(figsize=(8, 4))
    plt.hist(hav_km, bins=40, color="teal", alpha=0.8)
    plt.xlabel("Haversine distance (km)"); plt.ylabel("Count")
    plt.title("Geodesic error distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, "haversine_hist.png"))

    print("Done with evaluation!")

if __name__ == "__main__":
    main()