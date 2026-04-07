"""
Plot mIoU / core_mIoU / accuracy curves from train.log
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "train.log",
)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "viz")
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES_19 = [
    "wall", "floor_ground", "ceiling", "roof", "beam",
    "column", "window", "door_entrance", "stairs", "railing_fence",
    "balcony_corridor_canopy", "molding_parapet_buttress", "tower_chimney_dome",
    "furniture_object", "vegetation_vehicle", "garage", "roof_detail", "pool", "other",
]
# 10 core classes used in S3DIS standard benchmark
CORE_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def parse_log(path):
    """Parse train.log → list of dicts with epoch metrics."""
    records = []
    rec_epoch = None
    miou = cmiou = macc = allacc = None
    class_iou = {}

    # Regex for Val result line, e.g.:
    val_pat = re.compile(
        r"Val result: mIoU/core_mIoU/mAcc/allAcc\s+([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\."
    )
    # Regex for Class_N Result line, e.g.:
    cls_pat = re.compile(r"Class_(\d+)-[^\s]+\s+Result: iou/accuracy\s+([\d.]+)/([\d.]+)")
    # Train line: Train: [245/400][1/1693] ...
    train_pat = re.compile(r"Train:\s+\[(\d+)/\d+\]")

    cur_epoch = None

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = train_pat.search(line)
            if m:
                cur_epoch = int(m.group(1))

            m = val_pat.search(line)
            if m:
                miou = float(m.group(1))
                cmiou = float(m.group(2))
                macc = float(m.group(3))
                allacc = float(m.group(4))
                rec_epoch = cur_epoch

            m = cls_pat.search(line)
            if m:
                cid = int(m.group(1))
                class_iou[cid] = float(m.group(2))

            # A new training epoch starts → save previous Val record
            # (only if we have a complete record from a previous epoch)
            if rec_epoch is not None and cur_epoch != rec_epoch:
                if miou is not None:
                    records.append({
                        "epoch": rec_epoch,
                        "mIoU": miou, "core_mIoU": cmiou,
                        "mAcc": macc, "allAcc": allacc,
                        "class_iou": dict(class_iou),
                    })
                # reset for next block
                miou = cmiou = macc = allacc = None
                class_iou = {}
                rec_epoch = None

        # Flush last record at EOF
        if rec_epoch is not None and miou is not None:
            records.append({
                "epoch": rec_epoch,
                "mIoU": miou, "core_mIoU": cmiou,
                "mAcc": macc, "allAcc": allacc,
                "class_iou": dict(class_iou),
            })

    return records


def plot_overall_curves(records, save_path):
    """Left: mIoU / core_mIoU | Right: mAcc / allAcc"""
    epochs = [r["epoch"] for r in records]
    miou   = [r["mIoU"]   for r in records]
    cmiou  = [r["core_mIoU"] for r in records]
    macc   = [r["mAcc"]   for r in records]
    allacc = [r["allAcc"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # Left: IoU curves
    ax = axes[0]
    ax.plot(epochs, miou,  "b-",  linewidth=1.5, label="mIoU (19-class)")
    ax.plot(epochs, cmiou, "r--", linewidth=1.5, label="core_mIoU (10-class)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("mIoU / core_mIoU vs Epoch", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Right: Accuracy curves
    ax = axes[1]
    ax.plot(epochs, macc,   "g-",  linewidth=1.5, label="mAcc (19-class)")
    ax.plot(epochs, allacc, "m--", linewidth=1.5, label="allAcc")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("mAcc / allAcc vs Epoch", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    fig.suptitle("Training Validation Curves — PT-v3m2 Joint (S3DIS+ScanNet+BuildingNet)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


def plot_core_class_curves(records, core_ids, save_path):
    """10 core class IoU curves (S3DIS standard)."""
    n_cls = len(core_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    epochs = [r["epoch"] for r in records]

    for i, cid in enumerate(core_ids):
        curve = [r["class_iou"].get(cid, 0.0) for r in records]
        name = CLASS_NAMES_19[cid] if cid < len(CLASS_NAMES_19) else f"class_{cid}"
        ax.plot(epochs, curve, color=colors[i], linewidth=1.2,
                label=f"{cid}-{name}", alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("Per-Class IoU Curves — 10 Core Classes (S3DIS Benchmark)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)

    fig.suptitle("PT-v3m2 Joint Training — Core Class IoU Evolution",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_bars(records, core_ids, save_path):
    """Bar chart: best / final core class IoU."""
    last = records[-1]
    best_epoch = max(records, key=lambda r: r["core_mIoU"])

    names = [CLASS_NAMES_19[cid] for cid in core_ids]
    x = np.arange(len(core_ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    bars1 = ax.bar(x - w/2, [last["class_iou"].get(cid, 0) for cid in core_ids],
                   w, label=f"Final (epoch {last['epoch']})", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + w/2, [best_epoch["class_iou"].get(cid, 0) for cid in core_ids],
                   w, label=f"Best (epoch {best_epoch['epoch']}, core_mIoU={best_epoch['core_mIoU']:.3f})",
                   color="coral", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("10 Core Classes — Final vs Best IoU", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    records = parse_log(LOG_PATH)
    print(f"Parsed {len(records)} evaluation records (epochs: "
          f"{records[0]['epoch']} – {records[-1]['epoch']})")

    plot_overall_curves(
        records,
        os.path.join(SAVE_DIR, "training_curves_miou_acc.png"),
    )
    plot_core_class_curves(
        records,
        CORE_CLASS_IDS,
        os.path.join(SAVE_DIR, "training_curves_core_classes.png"),
    )
    plot_summary_bars(
        records,
        CORE_CLASS_IDS,
        os.path.join(SAVE_DIR, "training_curves_core_classes_bar.png"),
    )
    print("Done.")