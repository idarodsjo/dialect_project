
#!/usr/bin/env python3
"""
Plot eval accuracy and loss from one or more Hugging Face trainer_state.json files,
using absolute file paths defined inside this script (no CLI args).

Single figure with twin y-axes:
- Left  Y-axis: Eval Accuracy (combined across files)
- Right Y-axis: Eval Loss (combined across files)

Author: Ida's plotting helper
"""

import json
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# =========================
# ======== CONFIG =========
# =========================

# 🔧 Define your absolute paths to trainer_state.json files here:
JSON_PATHS: List[str] = [
    # Replace these with your actual files
    "/home/idatro/repo/did_prosody_whisper-main/model_output/nordia_unmod/checkpoint-8/trainer_state.json",
    "/home/idatro/repo/did_prosody_whisper-main/model_output/nordia_unmod/checkpoint-17/trainer_state.json",
    "/home/idatro/repo/did_prosody_whisper-main/model_output/nordia_unmod/checkpoint-24/trainer_state.json",
]

# Output filenames
OUTPUT_PLOT = "metrics_single_figure.png"   # PNG file for single-figure plot
OUTPUT_CSV  = None                          # e.g., "metrics.csv" to enable CSV export, or None to disable

# Plot appearance
plt.style.use("ggplot")
ACCURACY_YLIM = (0.0, 1.0)  # Set to None to auto-scale, or (0,1) if your accuracy is in [0,1]
LOSS_YSCALE   = None        # Set to "log" for log scale on loss axis, or None for linear

# =========================
# ====== FUNCTIONS ========
# =========================

def load_trainer_state(path: str) -> Dict[str, Any]:
    """Load a single trainer_state.json file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_log_entries(state: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    """
    Extract eval metrics from log_history entries.
    Expected keys: step, epoch, eval_accuracy, eval_loss.
    """
    logs = state.get("log_history", [])
    entries: List[Dict[str, Any]] = []
    for entry in logs:
        step = entry.get("step", entry.get("global_step"))
        epoch = entry.get("epoch")
        acc = entry.get("eval_accuracy")
        loss = entry.get("eval_loss")

        # Include entries that have at least step or epoch; metrics may be None
        if step is None and epoch is None:
            continue

        entries.append({
            "file": source_file,
            "step": step,
            "epoch": epoch,
            "eval_accuracy": acc,
            "eval_loss": loss,
        })
    return entries

def consolidate_logs(paths: List[str]) -> List[Dict[str, Any]]:
    """Load and merge entries from multiple trainer_state.json files."""
    all_entries: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] File not found, skipping: {p}")
            continue
        try:
            state = load_trainer_state(p)
            file_entries = extract_log_entries(state, source_file=p)
            all_entries.extend(file_entries)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")

    # Sort chronologically by step, then epoch
    def sort_key(e):
        s = e.get("step")
        ep = e.get("epoch")
        s = float(s) if s is not None else float("inf")
        ep = float(ep) if ep is not None else float("inf")
        return (s, ep)

    all_entries.sort(key=sort_key)
    return all_entries

def to_csv(entries: List[Dict[str, Any]], csv_path: str):
    import csv
    cols = ["file", "step", "epoch", "eval_accuracy", "eval_loss"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k) for k in cols})
    print(f"[INFO] Saved CSV: {csv_path}")

def plot_metrics_single_figure(entries: List[Dict[str, Any]], out_path: str):
    if not entries:
        print("[ERROR] No entries to plot.")
        return

    # Prepare combined series
    steps  = [e.get("step") for e in entries]
    epochs = [e.get("epoch") for e in entries]
    accs   = [e.get("eval_accuracy") for e in entries]
    losses = [e.get("eval_loss") for e in entries]

    # Filter for valid points (combined)
    steps_acc  = [s for s, a in zip(steps, accs)   if s is not None and a is not None]
    accs_f     = [a for s, a in zip(steps, accs)   if s is not None and a is not None]
    steps_loss = [s for s, l in zip(steps, losses) if s is not None and l is not None]
    losses_f   = [l for s, l in zip(steps, losses) if s is not None and l is not None]

    # Build figure with twin y-axes
    fig, ax_acc = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax_loss = ax_acc.twinx()

    # Combined curves only
    acc_line,   = ax_acc.plot(steps_acc,  accs_f,   color="#1f77b4", marker="o", label="Accuracy")
    loss_line,  = ax_loss.plot(steps_loss, losses_f, color="#d62728", marker="s", label="Loss")

    # Axes labels & title
    ax_acc.set_title("Validation Accuracy & Loss vs. Global Step and Epoch)")
    ax_acc.set_xlabel("Global Step")
    ax_acc.set_ylabel("Accuracy", color="#1f77b4")
    if ACCURACY_YLIM is not None:
        ax_acc.set_ylim(*ACCURACY_YLIM)
    ax_acc.tick_params(axis="y", colors="#1f77b4")

    ax_loss.set_ylabel("Loss", color="#d62728")
    if LOSS_YSCALE:
        ax_loss.set_yscale(LOSS_YSCALE)
    ax_loss.tick_params(axis="y", colors="#d62728")
    ax_acc.grid(True, alpha=0.3)

    # Secondary x-axis: sparse epoch ticks (aligned to accuracy points)
    epochs_acc = [ep for ep, a, s in zip(epochs, accs, steps) if ep is not None and a is not None and s is not None]
    if len(epochs_acc) == len(steps_acc) and len(steps_acc) > 1:
        ax_epoch = ax_acc.twiny()
        ax_epoch.set_xlim(ax_acc.get_xlim())
        tick_idx = list(range(0, len(steps_acc), max(1, len(steps_acc)//5)))
        epoch_ticks  = [steps_acc[i] for i in tick_idx]
        epoch_labels = [f"{epochs_acc[i]:.2f}" for i in tick_idx]
        ax_epoch.set_xticks(epoch_ticks)
        ax_epoch.set_xticklabels(epoch_labels)
        ax_epoch.set_xlabel("Epoch (at corresponding step)")

    # Unified legend with the two lines
    handles = [acc_line, loss_line]
    labels  = [h.get_label() for h in handles]
    ax_acc.legend(handles, labels, loc="best", fontsize=9)

    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved plot: {out_path}")

# =========================
# ========= MAIN ==========
# =========================

def main():
    # Validate input files
    valid_paths = [p for p in JSON_PATHS if isinstance(p, str) and os.path.isabs(p)]
    if not valid_paths:
        print("[ERROR] No absolute JSON paths defined. Edit JSON_PATHS in the script.")
        return

    print(f"[INFO] Using {len(valid_paths)} file(s):")
    for p in valid_paths:
        print(f"  - {p}")

    entries = consolidate_logs(valid_paths)
    if not entries:
        print("[ERROR] No log_history entries with eval metrics found.")
        return

    # Optional CSV export
    if OUTPUT_CSV:
        to_csv(entries, OUTPUT_CSV)

    # Single-figure plot (combined lines only)
    plot_metrics_single_figure(entries, OUTPUT_PLOT)

if __name__ == "__main__":
    main()