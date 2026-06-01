
#!/usr/bin/env python
# coding=utf-8
"""
Multitask training script for dialect classification + coordinate regression.

Design goals:
- Keep the encoder and the *classification* head identical to the baseline
  AutoModelForAudioClassification path (for fair comparison).
- Add a *parallel* regression head that reads the same pooled encoder representation
  and contributes to the total loss (so it backpropagates into the encoder).
- Do *not* modify the original model's forward; instead, override Trainer.compute_loss
  to add the regression loss (CE + lambda * MSE by default).
- Save the regression head parameters separately (regression_head.pt) alongside the
  regular save_pretrained() output, so evaluation can restore them.

Assumptions:
- The training/validation pickles contain at least the audio column, the label column,
  and the coordinate columns: latitude and longitude in *degrees*.
- Audio preprocessing (sampling rate, chunk length, padding/truncation) and all
  training hyperparameters remain identical to the classification-only baseline.

Normalization for coordinates (applied to targets before loss):
    lat_norm = latitude_deg / 90.0   (in [-1, 1])
    lon_norm = longitude_deg / 180.0 (in [-1, 1])
The same constants are used in evaluation for de/normalization.
"""
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datasets
import evaluate

from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

# --- Normalization constants shared with evaluation ---
LAT_DIVISOR = 90.0
LON_DIVISOR = 180.0


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-pool over the time dimension. If attention_mask is provided, use masked mean.
    last_hidden_state: (B, T, D); attention_mask: (B, T)
    Returns: (B, D)
    """
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    # Ensure mask is float and shape matches time steps
    mask = attention_mask.to(last_hidden_state.dtype)
    # Some models up/downsample time; if shapes mismatch, fall back to simple mean
    if mask.ndim != 2 or mask.shape[1] != last_hidden_state.shape[1]:
        return last_hidden_state.mean(dim=1)
    masked = last_hidden_state * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return masked.sum(dim=1) / denom


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Optional: name of dataset from `datasets` hub (unused here)"}
    )
    dataset_config_name: Optional[str] = field(default=None)

    train_file: Optional[str] = field(
        default=None, metadata={"help": "Pickle file containing training metadata and audio paths."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "Pickle file containing validation metadata and audio paths."}
    )

    train_split_name: str = field(default="train")
    eval_split_name: str = field(default="validation")

    audio_column_name: str = field(
        default="audio", metadata={"help": "Column containing audio path or array. Will be cast to Audio."}
    )
    label_column_name: str = field(
        default="label", metadata={"help": "Column containing classification labels (string)."}
    )

    latitude_column: str = field(default="latitude")
    longitude_column: str = field(default="longitude")

    offset_start_column: str = field(default="offset_start_ms")
    offset_end_column: str = field(default="offset_end_ms")

    max_length_seconds: float = field(default=30.0)

    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="NbAiLab/nb-whisper-medium",
        metadata={"help": "Pretrained model ID or path for AutoModelForAudioClassification"},
    )
    config_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")

    feature_extractor_name: Optional[str] = field(default=None)

    freeze_feature_encoder: bool = field(default=False)
    attention_mask: bool = field(default=False)

    use_auth_token: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)

    # Multitask
    regression_weight: float = field(default=1.0, metadata={"help": "Lambda for regression loss in total loss."})


def build_datasets(data_args: DataTrainingArguments, feature_extractor) -> DatasetDict:
    raw = DatasetDict()
    if data_args.train_file:
        train_df = pd.read_pickle(data_args.train_file)
        raw['train'] = Dataset.from_pandas(train_df)
    if data_args.eval_file:
        eval_df = pd.read_pickle(data_args.eval_file)
        raw['eval'] = Dataset.from_pandas(eval_df)

    # Cast label column to ClassLabel based on train set unique labels
    labels = sorted(list(set(raw['train'][data_args.label_column_name])))
    raw = raw.cast_column(data_args.label_column_name, ClassLabel(names=labels))

    # Ensure audio column exists
    for split in raw.keys():
        if data_args.audio_column_name not in raw[split].column_names:
            raise ValueError(
                f"--audio_column_name {data_args.audio_column_name} not found in {split} dataset. "
                f"Available columns: {raw[split].column_names}"
            )

    # Cast audio column to the extractor sampling rate
    raw = raw.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    return raw


def make_transforms(data_args: DataTrainingArguments, feature_extractor, model_input_name: str):
    target_sr = feature_extractor.sampling_rate
    max_len = int(round(target_sr * data_args.max_length_seconds))

    def _slice_with_offsets(batch: Dict[str, Any]) -> Dict[str, Any]:
        arrays = []
        labels = list(batch[data_args.label_column_name])
        lats = batch.get(data_args.latitude_column, None)
        lons = batch.get(data_args.longitude_column, None)
        starts = batch.get(data_args.offset_start_column, [0.0] * len(labels))
        ends = batch.get(data_args.offset_end_column, [None] * len(labels))

        # gather audio arrays
        for audio, s, e in zip(batch[data_args.audio_column_name], starts, ends):
            wav = audio['array']
            sr = audio['sampling_rate']
            # convert ms -> s if keys end with _ms
            s_sec = float(s) * (1e-3 if isinstance(s, (float, int)) and str(data_args.offset_start_column).endswith('_ms') else 1.0)
            if e is None:
                e_sec = min(s_sec + data_args.max_length_seconds, len(wav) / sr)
            else:
                e_sec = float(e) * (1e-3 if str(data_args.offset_end_column).endswith('_ms') else 1.0)
            start_frame = max(0, int(round(s_sec * sr)))
            end_frame = min(len(wav), int(round(e_sec * sr)))
            seg = wav[start_frame:end_frame]
            if seg.size == 0:
                seg = wav[: int(sr * data_args.max_length_seconds)]
            arrays.append(seg)

        # feature extractor handles resampling/pad/truncate
        inputs = feature_extractor(
            arrays,
            sampling_rate=target_sr,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,  # always request; masking use depends on model_args.attention_mask
        )
        out = {model_input_name: inputs.get(model_input_name)}
        # classification labels
        out['labels'] = labels
        # regression targets (normalized)
        if lats is None or lons is None:
            raise ValueError(
                f"Latitude/Longitude columns not found: '{data_args.latitude_column}', '{data_args.longitude_column}'"
            )
        lat_arr = np.asarray(lats, dtype=np.float32) / LAT_DIVISOR
        lon_arr = np.asarray(lons, dtype=np.float32) / LON_DIVISOR
        reg = np.stack([lat_arr, lon_arr], axis=1)
        out['regression_targets'] = reg
        return out

    return _slice_with_offsets


def get_hidden_size_from_model(model: nn.Module) -> int:
    # If the model exposes an encoder module directly (rare on WhisperForAudioClassification), try its first layer
    enc = getattr(model, "encoder", None)
    if enc and hasattr(enc, "layers") and len(enc.layers) > 0:
        kproj = getattr(enc.layers[0].self_attn, "k_proj", None)
        if isinstance(kproj, nn.Linear):
            return kproj.in_features

    # 1) Whisper: use config.d_model directly (no .model attribute on WhisperForAudioClassification)
    model_type = getattr(model.config, "model_type", None)
    if model_type == "whisper":
        d = getattr(model.config, "d_model", None)
        if isinstance(d, int) and d > 0:
            return d

    # 2) Try common classifier attributes (works for wav2vec2/ast/etc. when not Whisper)
    clf = getattr(model, 'classifier', None)
    if isinstance(clf, nn.Linear):
        return clf.in_features
    if hasattr(clf, 'out_proj') and isinstance(clf.out_proj, nn.Linear):
        return clf.out_proj.in_features

    # 3) Config fallbacks
    for key in ['hidden_size', 'd_model', 'proj_dim']:
        if hasattr(model.config, key):
            val = getattr(model.config, key)
            if isinstance(val, int) and val > 0:
                return val

    # 4) Named module fallbacks that appear in some audio models
    proj = getattr(model, 'projector', None)
    if isinstance(proj, nn.Linear):
        return proj.in_features

    raise ValueError("Could not infer encoder hidden size for regression head.")



class MultiTaskTrainer(Trainer):
    """Trainer that adds a regression head loss on top of the classification loss.
    The wrapped model is the original AutoModelForAudioClassification. We attach
    a small regression head module to `self.model.regression_head` and compute
    the combined loss inside `compute_loss`.
    """
    def __init__(self, *args, regression_weight: float = 1.0, use_masked_pool: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.regression_weight = float(regression_weight)
        self.use_masked_pool = bool(use_masked_pool)
        self.label_names = ["labels", "regression_targets"]

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            """
            Return:
            predictions = (logits, coords_pred)
            label_ids    = (labels, reg_targets)
            so compute_metrics can evaluate both tasks.
            """
            inputs = self._prepare_inputs(inputs)
            has_labels = all(k in inputs and inputs[k] is not None for k in self.label_names)

            # keep labels for compute_metrics
            labels_tuple = None
            if has_labels:
                labels_tuple = (inputs["labels"], inputs["regression_targets"])

            with torch.no_grad():
                if has_labels:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach()
                else:
                    loss = None
                    outputs = model(**inputs, return_dict=True)

                logits = outputs.get("logits", None)
                if logits is not None:
                    logits = logits.detach()

                # Regression prediction uses your existing pooled-cache mechanism
                hf_model = getattr(model, "module", model)
                pooled = getattr(hf_model, "_pooled_cache", None)
                coords_pred = None
                if pooled is not None and hasattr(hf_model, "regression_head"):
                    coords_pred = hf_model.regression_head(pooled).detach()

            if prediction_loss_only:
                return (loss, None, None)

            return (loss, (logits, coords_pred), labels_tuple)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        import torch
        import torch.nn as nn

        # ---------- Pull out fields that shouldn't go into model.forward ----------
        labels = inputs.pop("labels", None)
        reg_targets = inputs.pop("regression_targets", None)
        # We don't use attention_mask when hooking the classifier input
        inputs.pop("attention_mask", None)

        # ---------- Forward (no hidden states) ----------
        outputs = model(**inputs, return_dict=True)
        logits = outputs.get("logits", None)

        # ---------- Classification loss ----------
        loss_cls = torch.zeros((), device=logits.device, dtype=logits.dtype) if logits is not None else 0.0
        if labels is not None and logits is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=logits.device, dtype=torch.long)
            else:
                labels = labels.to(logits.device).long()

            num_labels = logits.size(-1)
            loss_cls = nn.CrossEntropyLoss()(logits.view(-1, num_labels), labels.view(-1))

        # ---------- Get pooled features captured by the hook ----------
        hf_model = getattr(model, "module", model)  # unwrap if DeepSpeed/DP wrapped
        pooled = getattr(hf_model, "_pooled_cache", None)
        if pooled is None:
            raise RuntimeError(
                "Classifier pre-hook did not capture pooled features. "
                "Ensure the hook is registered on `model.classifier` before training."
            )

        # ---------- Regression head & loss ----------
        regression_head = getattr(hf_model, "regression_head", None)
        if regression_head is None:
            raise RuntimeError("model.regression_head is missing; attach it before training.")

        coords_pred = regression_head(pooled)  # (B, 2)
        loss_reg = torch.zeros((), device=coords_pred.device, dtype=coords_pred.dtype)
        if reg_targets is not None:
            if not isinstance(reg_targets, torch.Tensor):
                reg_targets = torch.tensor(reg_targets, device=coords_pred.device, dtype=coords_pred.dtype)
            else:
                reg_targets = reg_targets.to(coords_pred.device, dtype=coords_pred.dtype)
            loss_reg = nn.MSELoss()(coords_pred, reg_targets)

        # ---------- Total loss ----------
        reg_w_t = coords_pred.new_tensor(getattr(self, "regression_weight", 1.0))
        loss = (1.0 - reg_w_t) * loss_cls + reg_w_t * loss_reg

        # ---------- Logging ----------
        self.log({"loss_cls": float(loss_cls.detach()), "loss_reg": float(loss_reg.detach())})

        return (loss, outputs) if return_outputs else loss



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)
    logger.info("Training/evaluation parameters %s", training_args)

    # Seed
    set_seed(training_args.seed)

    # Check output dir
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the output_dir or add --overwrite_output_dir."
            )

    # Feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Datasets
    raw_datasets = build_datasets(data_args, feature_extractor)

    # Model input name (e.g., "input_values" or "input_features")
    model_input_name = feature_extractor.model_input_names[0]

    # Transforms
    transform_fn = make_transforms(data_args, feature_extractor, model_input_name)
    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets['train'] = raw_datasets['train'].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        raw_datasets['train'].set_transform(transform_fn, output_all_columns=False)
    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets['eval'] = raw_datasets['eval'].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
        raw_datasets['eval'].set_transform(transform_fn, output_all_columns=False)

    # Labels mapping
    labels = raw_datasets['train'].features[data_args.label_column_name].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Config and model
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Ensure hidden states are returned when asked
    config.output_hidden_states = False

    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    model._pooled_cache = None

    def _store_pooled_input(module, inputs):
        # inputs is a tuple; take the tensor going into the classifier
        model._pooled_cache = inputs[0]  # (B, D) with gradient

    # Keep handle if you want to remove later
    model._clf_hook = model.classifier.register_forward_pre_hook(lambda m, i: _store_pooled_input(m, i))

    if model_args.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()


    # Dimension going into the final classifier (works with the pre-hook approach)
    if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
        clf_in = model.classifier.in_features
    else:
        # Fallback for unusual heads
        clf_in = get_hidden_size_from_model(model)
    dropout_p = getattr(model.config, 'classifier_dropout_prob', None) or getattr(model.config, 'hidden_dropout_prob', None) or 0.1
    model.regression_head = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(clf_in, 64), # HIDDEN LAYER = 64
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(64, 64),  # second hidden layer
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(64, 2),
    )


    # --- Capture pooled features that go into the classifier (B, D) ---
    hf_model = model  # not yet wrapped by DeepSpeed/Accelerate here
    hf_model._pooled_cache = None

    def _store_classifier_input(module, inputs):
        # inputs is a tuple; first element is the tensor fed into the classifier
        # shape: (batch_size, hidden_dim)
        hf_model._pooled_cache = inputs[0]

    # Register a forward-pre-hook on the final classifier
    if not hasattr(hf_model, "classifier"):
        raise RuntimeError("Expected `model.classifier` to exist on the audio classification model.")
    hf_model._clf_hook = hf_model.classifier.register_forward_pre_hook(_store_classifier_input)

    
    import numpy as np
    import evaluate

    metric_acc = evaluate.load("accuracy")

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0088
        lat1 = np.radians(lat1); lon1 = np.radians(lon1)
        lat2 = np.radians(lat2); lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    def stable_ce_from_logits(logits, labels):
        # CE that matches torch.nn.CrossEntropyLoss reduction='mean'
        logits = logits - logits.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        log_probs = logits - logsumexp
        return float(-np.mean(log_probs[np.arange(labels.shape[0]), labels]))

    def make_compute_metrics(regression_weight: float, tau_km: float = 100.0):
        w = float(regression_weight)

        def compute_metrics(eval_pred):
            (logits, coords_pred) = eval_pred.predictions
            (labels, reg_targets) = eval_pred.label_ids

            # ---- Classification: accuracy + CE ----
            pred_ids = np.argmax(logits, axis=1)
            acc = metric_acc.compute(predictions=pred_ids, references=labels)["accuracy"]
            ce = stable_ce_from_logits(logits, labels)

            # ---- Regression: normalized MSE + geodesic km ----
            # coords are normalized in your pipeline (lat/90, lon/180) [1](placeholder-0)
            mse_norm = float(np.mean((coords_pred - reg_targets) ** 2))

            lat_true = reg_targets[:, 0] * LAT_DIVISOR
            lon_true = reg_targets[:, 1] * LON_DIVISOR
            lat_pred = coords_pred[:, 0] * LAT_DIVISOR
            lon_pred = coords_pred[:, 1] * LON_DIVISOR

            # keep within valid ranges (optional but helps stability)
            lat_pred = np.clip(lat_pred, -90.0, 90.0)
            lon_pred = ((lon_pred + 180.0) % 360.0) - 180.0

            dist_km = haversine_km(lat_true, lon_true, lat_pred, lon_pred)
            geo_mean_km = float(np.mean(dist_km))
            geo_median_km = float(np.median(dist_km))

            # Convert regression error to a [0,1] score so it can be mixed with accuracy
            geo_score = float(np.exp(-geo_mean_km / float(tau_km)))

            # ---- Objectives with weighting w ----
            # (A) score objective (higher is better) matches your requested behavior
            obj_score = (1.0 - w) * acc + w * geo_score

            # (B) loss objective (lower is better) mirrors your training mixture exactly [1](placeholder-0)
            obj_loss = (1.0 - w) * ce + w * mse_norm

            # Optional: if you truly want “only regression metrics” at w=1 and “only acc” at w=0,
            # obj_score already does that automatically because weights become 0/1.

            return {
                # individual task metrics (always useful to log)
                "accuracy": float(acc),
                "ce": float(ce),
                "reg_mse_norm": float(mse_norm),
                "geo_mean_km": geo_mean_km,
                "geo_median_km": geo_median_km,
                "geo_score": geo_score,

                # weighted objectives
                "obj_score": float(obj_score),  # higher better
                "obj_loss": float(obj_loss),    # lower better
            }

        return compute_metrics

    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets['train'] if training_args.do_train else None,
        eval_dataset=raw_datasets['eval'] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        compute_metrics=make_compute_metrics(regression_weight=model_args.regression_weight,tau_km=100.0),
        regression_weight=model_args.regression_weight,
        use_masked_pool=model_args.attention_mask,
    )

    # Train
    if training_args.do_train:
        ckpt = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=ckpt)
        trainer.save_model()  # saves the HF model
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Also save the regression head separately for easy restoration in eval
        out_dir = training_args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.regression_head.state_dict(), os.path.join(out_dir, 'regression_head.pt'))
        with open(os.path.join(out_dir, 'multitask_meta.json'), 'w') as f:
            json.dump({
                "regression_weight": model_args.regression_weight,
                "latitude_divisor": LAT_DIVISOR,
                "longitude_divisor": LON_DIVISOR,
                "latitude_column": data_args.latitude_column,
                "longitude_column": data_args.longitude_column,
            }, f, indent=2)

    # Eval
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
