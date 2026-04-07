"""
Segmentation Visualization Script
Compare ground truth vs predicted segmentation on S3DIS and BuildingNet test sets.
Generates before/after comparison figures and detailed close-up views.

Usage (run on Ubuntu with CUDA):
    cd /home/yang/PointCloud_Datasets
    python /mnt/c/Yang/Pointcept-main/Pointcept-main/exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet/code/viz_semseg_comparison.py

Output images saved to:
    /mnt/c/Yang/Pointcept-main/Pointcept-main/exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet/viz/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# ── Project paths ──────────────────────────────────────────────────────────────
EXP_DIR = "/mnt/c/Yang/Pointcept-main/Pointcept-main/exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet"
CODE_DIR = os.path.join(EXP_DIR, "code")
MODEL_PATH = os.path.join(EXP_DIR, "model", "model_best.pth")
VIZ_DIR = os.path.join(EXP_DIR, "viz")

S3DIS_DATA_ROOT = "/home/yang/PointCloud_Datasets/S3DIS_processed"
BUILDNET_DATA_ROOT = "/home/yang/PointCloud_Datasets/BuildingNet_processed"

# ── Unified class names (19 classes) ────────────────────────────────────────
CLASS_NAMES = [
    "wall", "floor_ground", "ceiling", "roof", "beam", "column",
    "window", "door_entrance", "stairs", "railing_fence",
    "balcony_corridor_canopy", "molding_parapet_buttress",
    "tower_chimney_dome", "furniture_object",
    "vegetation_vehicle", "garage", "roof_detail", "pool", "other",
]

# ── S3DIS remap (13 → 19 unified) ────────────────────────────────────────────
S3DIS_REMAP = [2, 1, 0, 4, 5, 6, 7, 13, 13, 13, 13, 13, 18]

# ── BuildingNet remap (30 → 19 unified) ──────────────────────────────────────
BUILD_REMAP = [
    0, 6, 14, 3, 14, 7, 12, 13, 1, 4, 8, 5, 9, 12, 12, 15,
    9, 16, 10, 10, 11, 7, 10, 17, 16, 11, 11, 6, 16, 18, -1, -1,
]

# ── Color palette for 19 classes ───────────────────────────────────────────────
PALETTE = [
    "#e6194b",  # 0  wall
    "#3cb44b",  # 1  floor_ground
    "#ffe119",  # 2  ceiling
    "#4363d8",  # 3  roof
    "#f58231",  # 4  beam
    "#911eb4",  # 5  column
    "#46f0f0",  # 6  window
    "#f032e6",  # 7  door_entrance
    "#bcf60c",  # 8  stairs
    "#fabebe",  # 9  railing_fence
    "#008080",  # 10 balcony_canopy
    "#e6beff",  # 11 molding
    "#9a6324",  # 12 tower_chimney
    "#808080",  # 13 furniture_object
    "#800000",  # 14 vegetation_vehicle
    "#aaffc3",  # 15 garage
    "#000075",  # 16 roof_detail
    "#808000",  # 17 pool
    "#ffffff",  # 18 other
]


def class_to_rgb(labels, palette=PALETTE):
    """Map class IDs → RGB float array (0-1)."""
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    for i, l in enumerate(labels):
        c = int(l)
        if 0 <= c < len(palette):
            colors[i] = mcolors.to_rgb(palette[c])
        else:
            colors[i] = (0.5, 0.5, 0.5)
    return colors


def remap_labels(labels, remap):
    """Remap original class IDs to unified IDs."""
    out = np.full(labels.shape, -1, dtype=labels.dtype)
    for orig, unified in enumerate(remap):
        if unified >= 0:
            out[labels == orig] = unified
    return out


def save_comparison(coord, color_orig, labels_pred, labels_gt, save_path,
                    elev=30, azim=45, n_pts=20000, point_size=2):
    """Left: original RGB color.  Right: predicted segmentation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(coord)
    if n > n_pts:
        idx = np.random.choice(n, n_pts, replace=False)
    else:
        idx = np.arange(n)

    cs, co, cp = coord[idx], color_orig[idx], labels_pred[idx]
    # S3DIS color can be 0-255 or 0-1; clip to valid range for matplotlib
    co = np.clip(co, 0.0, 1.0)

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(cs[:, 0], cs[:, 1], cs[:, 2], c=co, s=point_size, alpha=0.85)
    ax1.set_title("Original Color", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_box_aspect(None, zoom=0.85)

    ax2 = fig.add_subplot(122, projection='3d')
    pred_rgb = class_to_rgb(cp)
    ax2.scatter(cs[:, 0], cs[:, 1], cs[:, 2], c=pred_rgb, s=point_size, alpha=0.85)
    ax2.set_title("Predicted Segmentation", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_box_aspect(None, zoom=0.85)

    present = np.unique(cp)
    legend = [
        Patch(facecolor=PALETTE[c], label=CLASS_NAMES[c])
        for c in present if 0 <= c < len(PALETTE)
    ]
    if legend:
        ax2.legend(handles=legend, loc='upper left', fontsize=6.5, ncol=1,
                   framealpha=0.8, bbox_to_anchor=(-0.05, 1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def save_gt_pred_comparison(coord, color_orig, labels_pred, labels_gt, save_path,
                             elev=30, azim=45, n_pts=20000, point_size=2):
    """3-column: original color | predicted | ground truth."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(coord)
    if n > n_pts:
        idx = np.random.choice(n, n_pts, replace=False)
    else:
        idx = np.arange(n)

    cs, co, cp, cg = coord[idx], color_orig[idx], labels_pred[idx], labels_gt[idx]
    co = np.clip(co, 0.0, 1.0)

    fig = plt.figure(figsize=(22, 7))
    configs = [
        (1, co, "Original Color"),
        (2, class_to_rgb(cp), "Predicted Segmentation"),
        (3, class_to_rgb(cg), "Ground Truth"),
    ]
    for col, rgb, title in configs:
        ax = fig.add_subplot(1, 3, col, projection='3d')
        ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2], c=rgb, s=point_size, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(None, zoom=0.85)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


def find_local_regions(coord, labels, target_classes, min_points=200):
    """Find bounding boxes around specific semantic regions.
    Returns list of (center, half_size) for each region.
    """
    regions = []
    for cls in target_classes:
        mask = labels == cls
        if mask.sum() < min_points:
            continue
        pts = coord[mask]
        c = pts.mean(axis=0)
        span = pts.max(axis=0) - pts.min(axis=0)
        half_size = span.max() * 0.6  # zoom to 60% of max span
        regions.append((c, half_size, cls))
    return regions


def crop_and_zoom(coord, color, pred, gt, center, half_size, margin=1.5):
    """Crop points within a box around center, zoom in by adjusting axis limits."""
    cx, cy, cz = center
    hs = half_size * margin
    mask = (
        (coord[:, 0] >= cx - hs) & (coord[:, 0] <= cx + hs) &
        (coord[:, 1] >= cy - hs) & (coord[:, 1] <= cy + hs) &
        (coord[:, 2] >= cz - hs) & (coord[:, 2] <= cz + hs)
    )
    if mask.sum() < 50:
        return None
    return coord[mask], color[mask], pred[mask], gt[mask]


def save_closeup_panels(coord, color_orig, labels_pred, labels_gt, scene_name,
                        dataset_name, save_dir,
                        target_classes=None, n_closeups=5,
                        n_pts=15000, point_size=3.5):
    """Generate zoomed close-up panels of specific semantic regions.

    For each detected region, shows a 2×3 multi-view grid:
      Row 1 (top): predicted segmentation — 6 different viewing angles
      Row 2 (bottom): ground truth — same 6 angles (paired)

    Angles:
      0. Top-down   (elev=80, azim=0)   — bird's-eye
      1. Front      (elev=15, azim=0)   — facing +Y
      2. Right side (elev=15, azim=90) — facing +X
      3. 45° corner (elev=30, azim=45)
      4. Back-diag  (elev=30, azim=225)
      5. Left side  (elev=15, azim=270)

    This lets you directly compare pred vs GT at matching viewpoints,
    which is essential for assessing boundary quality and 3-D structure.
    """
    os.makedirs(save_dir, exist_ok=True)

    n = len(coord)
    if n > n_pts:
        idx = np.random.choice(n, n_pts, replace=False)
    else:
        idx = np.arange(n)

    coord, color_o, pred, gt = coord[idx], color_orig[idx], labels_pred[idx], labels_gt[idx]

    if target_classes is None:
        target_classes = list(range(19))

    # Find regions
    regions = find_local_regions(coord, pred, target_classes, min_points=150)

    # Auto-select diverse regions if not enough found
    if len(regions) < n_closeups:
        for cls in range(19):
            mask = pred == cls
            if mask.sum() >= 100:
                pts = coord[mask]
                c = pts.mean(axis=0)
                span = pts.max(axis=0) - pts.min(axis=0)
                hs = span.max() * 0.5
                regions.append((c, hs, cls))
                if len(regions) >= n_closeups * 2:
                    break

    # Deduplicate & take top n_closeups
    selected = []
    for r in regions[:n_closeups * 3]:
        c, hs, cls = r
        too_close = any(np.linalg.norm(c - sc) < hs * 0.8 for sc, _, _ in selected)
        if not too_close:
            selected.append(r)
        if len(selected) >= n_closeups:
            break

    if not selected:
        print(f"  WARNING: no suitable regions found for {scene_name}")
        return

    # ── 6 interior viewing angles at eye-level ─────────────────────────────
    # Simulates standing INSIDE the room, turning head 360° at eye height.
    # elev=3 → nearly horizontal gaze; axis limits = scene bounding box →
    # camera is positioned at the center of the box, feeling "inside" the space.
    INTERIOR_ANGLES = [
        (3,   0, "Facing +Y  (0°)"),
        (3,  60, "Facing +Y+X (60°)"),
        (3, 120, "Facing +X  (120°)"),
        (3, 180, "Facing -Y  (180°)"),
        (3, 240, "Facing -Y-X (240°)"),
        (3, 300, "Facing -X  (300°)"),
    ]

    for ri, (center, half_size, cls_id) in enumerate(selected):
        crop = crop_and_zoom(coord, color_o, pred, gt, center, half_size, margin=2.0)
        if crop is None:
            continue
        c_coord, c_color, c_pred, c_gt = crop

        # Scene bounding box — used for interior-view axis limits
        scene_min = c_coord.min(axis=0)
        scene_max = c_coord.max(axis=0)
        xm, xM = scene_min[0], scene_max[0]
        ym, yM = scene_min[1], scene_max[1]
        zm, zM = scene_min[2], scene_max[2]

        # ── 2 rows × 3 cols ─────────────────────────────────────────────────
        # Col 0-2: pred vs GT for each group of 2 interior angles
        fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                                 subplot_kw={"projection": "3d"})
        fig.patch.set_facecolor("white")

        # Pair up 6 angles as (pred_row, col) → 2 per column
        angle_pairs = [(INTERIOR_ANGLES[i], INTERIOR_ANGLES[i + 3])
                       for i in range(3)]

        for col_idx, (a1, a2) in enumerate(angle_pairs):
            for row_idx, (labels, elev, azim, label) in enumerate([
                    (c_pred, *a1[:2], "Pred"),
                    (c_gt,   *a2[:2], "GT"),
            ]):
                ax = axes[row_idx, col_idx]
                rgb = class_to_rgb(labels)
                ax.scatter(
                    c_coord[:, 0], c_coord[:, 1], c_coord[:, 2],
                    c=rgb, s=point_size, alpha=0.85
                )
                ax.set_title(f"{label}: {a1[2] if row_idx == 0 else a2[2]}",
                             fontsize=10, fontweight='bold', pad=4)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlim(xm, xM)
                ax.set_ylim(ym, yM)
                ax.set_zlim(zm, zM)
                ax.set_box_aspect(None, zoom=1.4)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])

        class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        fig.suptitle(
            f"{scene_name}  ·  {dataset_name}\n"
            f"Region {ri+1}: {class_name}  ·  "
            f"Pred acc: {(c_pred[c_gt >= 0] == c_gt[c_gt >= 0]).mean():.1%}  ·  "
            f"{len(c_coord):,} pts  ·  Interior view (eye-level)",
            fontsize=15, fontweight='bold', y=0.98
        )
        plt.tight_layout(rect=[0.025, 0.03, 1, 0.95])

        out = os.path.join(
            save_dir,
            f"{scene_name}_closeup{ri+1}_{class_name}.png"
        )
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    sys.path.insert(0, CODE_DIR)
    from pointcept.models import build_model

    cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=19,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m2",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 12, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 96, 192, 384),
            dec_num_head=(4, 6, 12, 24),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0,
            drop_path=0.3, shuffle_orders=True,
            pre_norm=True, enable_rpe=False,
            enable_flash=True, upcast_attention=False,
            upcast_softmax=False, traceable=False,
            mask_token=False, enc_mode=False,
            freeze_encoder=False,
        ),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0,
                       ignore_index=-1, weight=[1.0]*19)],
    )

    model = build_model(cfg)
    return model


def load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v
    model.load_state_dict(new_sd, strict=True)
    print(f"  Loaded: {path}  (epoch {ckpt.get('epoch', '?')})")
    return model


def infer(model, data_dict, grid_size=0.02):
    """Fragment-based inference matching the test pipeline.

    Args:
        model: PT-v3m2 segmentor
        data_dict: dict with keys coord(N,3), color(N,3), segment(N,)
        grid_size: voxel size used in GridSample test transform

    Returns:
        pred (np.ndarray): predicted class ID for each input point (N,)
        pred_full (np.ndarray): prediction probabilities (N, 19)
    """
    import sys; sys.path.insert(0, CODE_DIR)
    from pointcept.datasets.transform import GridSample, CenterShift
    from copy import deepcopy

    model.eval()
    device = next(model.parameters()).device

    coord = data_dict["coord"].copy()          # (N, 3)
    color = data_dict["color"].copy()          # (N, 3)
    segment = data_dict["segment"].copy()      # (N,)

    # ── 1. CenterShift (apply_z=True) — same as config data.test.transform ──
    #     NOT coord.mean(): Pointcept uses bbox XY center + z_min floor shift.
    _d = {"coord": coord.astype(np.float32)}
    CenterShift(apply_z=True)(_d)
    coord = _d["coord"]

    # ── 2. NormalizeColor (scale to 0-1) ───────────────────────────────────
    if color.max() > 1.0:
        color = color / 255.0

    # ── 3. Build GridSample in test mode ──────────────────────────────────
    voxelize = GridSample(
        grid_size=grid_size,
        hash_type="fnv",
        mode="test",
        return_grid_coord=True,
    )

    data = dict(
        coord=coord.astype(np.float32),
        feat=color.astype(np.float32),
        index_valid_keys=["coord", "feat"],
    )
    fragments = voxelize(data)  # list of fragments

    # ── 4. Inference per fragment ─────────────────────────────────────────
    n_total = coord.shape[0]
    pred_prob = np.zeros((n_total, 19), dtype=np.float32)

    cs_frag = CenterShift(apply_z=False)

    for frag in fragments:
        frag = deepcopy(frag)
        idx_part = frag.pop("index")  # (M,) - indices into original point cloud
        gc = frag["grid_coord"]       # (M, 3)
        feat = frag["feat"]            # (M, 3)
        # ── Per-fragment CenterShift(apply_z=False) — test_cfg.post_transform ──
        _f = {"coord": frag["coord"].astype(np.float32).copy()}
        cs_frag(_f)
        coord_f = _f["coord"]

        # offset is cumulative point count per batch; single batch = [M]
        offset_np = np.array([coord_f.shape[0]], dtype=np.int32)
        batch_np = np.zeros(coord_f.shape[0], dtype=np.int32)  # single batch = all zeros

        # Rebuild data_dict in Point format
        input_dict = {
            "coord": torch.from_numpy(coord_f).float().cuda(),           # (M, 3)
            "feat": torch.from_numpy(np.concatenate([coord_f, feat], axis=1)).float().cuda(),  # (M, 6)
            "offset": torch.from_numpy(offset_np).long().cuda(),          # (1,)
            "batch": torch.from_numpy(batch_np).long().cuda(),           # (M,)
            "grid_coord": torch.from_numpy(gc).long().cuda(),             # (M, 3)
            "grid_size": grid_size,
        }

        # Debug: verify shapes
        assert input_dict["coord"].shape[0] == input_dict["batch"].shape[0], \
            f"coord ({input_dict['coord'].shape[0]}) != batch ({input_dict['batch'].shape[0]})"
        assert input_dict["coord"].shape[0] == input_dict["grid_coord"].shape[0], \
            f"coord ({input_dict['coord'].shape[0]}) != grid_coord ({input_dict['grid_coord'].shape[0]})"

        with torch.no_grad():
            out = model(input_dict)
            logits = out["seg_logits"]
            # DefaultSegmentorV2 / tester use shape (M, num_classes), not (1, M, C).
            # Using prob[0] would broadcast one row to all points → single-class collapse.
            if logits.dim() == 3:
                logits = logits[0]
            prob = F.softmax(logits.float(), dim=-1)
            pred_prob[idx_part] += prob.cpu().numpy()

    pred = pred_prob.argmax(axis=1)
    return pred, pred_prob


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_s3dis(data_root, split, room_idx):
    """Load raw S3DIS room: coord (N,3), color (N,3), seg_unified (N,).

    Pointcept 预处理后的标准结构（与你截图一致）::

        {data_root}/{split}/{房间名}/coord.npy, color.npy, segment.npy

    例如: S3DIS_processed/Area_5/WC_1/coord.npy
    """
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(split_path)

    room_dirs = sorted(
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    )
    # 标准层级：split 下每个子目录是一间房，内含 coord.npy
    if room_dirs:
        probe = os.path.join(split_path, room_dirs[0], "coord.npy")
        if os.path.isfile(probe):
            room = room_dirs[room_idx % len(room_dirs)]
            room_path = os.path.join(split_path, room)
            name = f"{split}-{room}"
            coord = np.load(os.path.join(room_path, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(room_path, "color.npy")).astype(np.float32)
            seg13 = np.load(os.path.join(room_path, "segment.npy")).reshape(-1)
            seg19 = remap_labels(seg13, S3DIS_REMAP)
            return coord, color, seg19, name

    # 备选：split 目录下无房间子目录，仅有 room_coord.npy / room_color.npy / room_segment.npy
    entries = os.listdir(split_path)
    npy = [e for e in entries if e.endswith(".npy")]
    npz = [e for e in entries if e.endswith(".npz")]
    if npy and not room_dirs:
        prefixes = sorted(
            p for p in set(e.rsplit("_", 1)[0] for e in npy if "_" in e)
            if all(
                os.path.isfile(os.path.join(split_path, f"{p}_{k}.npy"))
                for k in ("coord", "color", "segment")
            )
        )
        if prefixes:
            prefix = prefixes[room_idx % len(prefixes)]
            name = f"{split}-{prefix}"
            coord = np.load(os.path.join(split_path, f"{prefix}_coord.npy")).astype(np.float32)
            color = np.load(os.path.join(split_path, f"{prefix}_color.npy")).astype(np.float32)
            seg13 = np.load(os.path.join(split_path, f"{prefix}_segment.npy")).reshape(-1)
            seg19 = remap_labels(seg13, S3DIS_REMAP)
            return coord, color, seg19, name
    if npz:
        npz = sorted(npz)
        zf = npz[room_idx % len(npz)]
        data = np.load(os.path.join(split_path, zf))
        name = f"{split}-{os.path.splitext(zf)[0]}"
        seg19 = remap_labels(data["segment"].reshape(-1), S3DIS_REMAP)
        return data["coord"].astype(np.float32), data["color"].astype(np.float32), seg19, name

    raise FileNotFoundError(f"No valid S3DIS rooms under {split_path}")


def load_buildingnet(data_root, split, sample_idx):
    """Load raw BuildingNet sample: coord, color, seg_unified."""
    split_path = os.path.join(data_root, split)
    entries = os.listdir(split_path)
    # Support both directories with .npy files and flat .npz files
    dirs = sorted([e for e in entries if os.path.isdir(os.path.join(split_path, e))])
    npz_files = sorted([e for e in entries if e.endswith(".npz")])

    if dirs:
        names = dirs
        name = names[sample_idx % len(names)]
        sample_path = os.path.join(split_path, name)
        coord = np.load(os.path.join(sample_path, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(sample_path, "color.npy")).astype(np.float32)
        seg30 = np.load(os.path.join(sample_path, "segment.npy")).reshape(-1)
    elif npz_files:
        name = npz_files[sample_idx % len(npz_files)]
        sample_path = os.path.join(split_path, name)
        data = np.load(sample_path)
        coord = data["coord"].astype(np.float32)
        color = data["color"].astype(np.float32)
        seg30 = data["segment"].reshape(-1)
        name = os.path.splitext(name)[0]
    else:
        raise FileNotFoundError(f"No samples found in {split_path}")

    seg19 = remap_labels(seg30, BUILD_REMAP)
    return coord, color, seg19, name


def count_s3dis(data_root, split):
    """统计 split（如 Area_5）下的房间数，与 Pointcept 目录一致。"""
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        return 0
    room_dirs = sorted(
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    )
    if room_dirs and os.path.isfile(os.path.join(split_path, room_dirs[0], "coord.npy")):
        return len(room_dirs)
    entries = os.listdir(split_path)
    npy = [e for e in entries if e.endswith(".npy")]
    npz = [e for e in entries if e.endswith(".npz")]
    if npy and not room_dirs:
        prefixes = [
            p for p in sorted(set(e.rsplit("_", 1)[0] for e in npy if "_" in e))
            if all(
                os.path.isfile(os.path.join(split_path, f"{p}_{k}.npy"))
                for k in ("coord", "color", "segment")
            )
        ]
        return len(prefixes)
    if npz:
        return len(npz)
    return 0


def count_buildingnet(data_root, split):
    split_path = os.path.join(data_root, split)
    entries = os.listdir(split_path)
    dirs = [e for e in entries if os.path.isdir(os.path.join(split_path, e))]
    npz = [e for e in entries if e.endswith(".npz")]
    if dirs:
        return len(dirs)
    elif npz:
        return len(npz)
    else:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="S3DIS & BuildingNet segmentation visualization: "
                    "before (original color) vs after (predicted segmentation)"
    )
    parser.add_argument("--s3dis_n", type=int, default=2,
                        help="Number of S3DIS rooms to visualize")
    parser.add_argument("--buildingnet_n", type=int, default=2,
                        help="Number of BuildingNet samples to visualize")
    parser.add_argument("--s3dis_idx", type=int, nargs="+", default=[0, 5, 10],
                        help="S3DIS room indices (area-order sorted)")
    parser.add_argument("--buildingnet_idx", type=int, nargs="+", default=[0, 3, 6, 9],
                        help="BuildingNet sample indices (alphabetical)")
    parser.add_argument("--n_closeups", type=int, default=5,
                        help="Number of close-up panels per sample")
    parser.add_argument("--points", type=int, default=20000,
                        help="Max points per subplot (subsampling for speed)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to model_best.pth")
    args = parser.parse_args()

    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(os.path.join(VIZ_DIR, "s3dis"), exist_ok=True)
    os.makedirs(os.path.join(VIZ_DIR, "buildingnet"), exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print("\n[1/4] Building and loading model ...")
    model = build_model()
    model = load_weights(model, args.model_path)
    model = model.cuda()
    model.eval()
    print("  Model ready on CUDA.")

    # ── S3DIS ──────────────────────────────────────────────────────────────
    print("\n[2/4] S3DIS Area_5 visualization ...")
    n_rooms = count_s3dis(S3DIS_DATA_ROOT, "Area_5")
    print(f"  Total rooms in Area_5: {n_rooms}")

    for i, room_idx in enumerate(args.s3dis_idx[:args.s3dis_n]):
        print(f"\n  [{i+1}/{min(args.s3dis_n, len(args.s3dis_idx))}] "
              f"Loading room index {room_idx} ...")
        try:
            coord, color, seg_gt, name = load_s3dis(S3DIS_DATA_ROOT, "Area_5", room_idx)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        print(f"    Points: {len(coord)}, "
              f"GT classes: {sorted(np.unique(seg_gt).tolist())}")

        pred, _ = infer(model, {"coord": coord, "color": color, "segment": seg_gt},
                        grid_size=0.02)
        print(f"    Pred classes: {sorted(np.unique(pred).tolist())}")

        mask = seg_gt >= 0
        acc = (pred[mask] == seg_gt[mask]).mean()
        print(f"    Pixel accuracy: {acc:.3f}")

        out_s3dis = os.path.join(VIZ_DIR, "s3dis", name)
        save_comparison(coord, color, pred, seg_gt,
                        f"{out_s3dis}_comparison.png",
                        elev=30, azim=45, n_pts=args.points)
        save_gt_pred_comparison(coord, color, pred, seg_gt,
                                f"{out_s3dis}_gt_pred.png",
                                elev=30, azim=45, n_pts=args.points)
        # Close-up panels of local semantic regions
        save_closeup_panels(coord, color, pred, seg_gt, name, "S3DIS",
                            os.path.join(VIZ_DIR, "s3dis"),
                            n_closeups=args.n_closeups, n_pts=args.points)

    # ── BuildingNet ────────────────────────────────────────────────────────
    print("\n[3/4] BuildingNet test visualization ...")
    n_samples = count_buildingnet(BUILDNET_DATA_ROOT, "test")
    print(f"  Total samples in test: {n_samples}")

    for i, sample_idx in enumerate(args.buildingnet_idx[:args.buildingnet_n]):
        print(f"\n  [{i+1}/{min(args.buildingnet_n, len(args.buildingnet_idx))}] "
              f"Loading sample index {sample_idx} ...")
        try:
            coord, color, seg_gt, name = load_buildingnet(
                BUILDNET_DATA_ROOT, "test", sample_idx)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        print(f"    Points: {len(coord)}, "
              f"GT classes: {sorted(np.unique(seg_gt).tolist())}")

        pred, _ = infer(model, {"coord": coord, "color": color, "segment": seg_gt},
                        grid_size=0.05)
        print(f"    Pred classes: {sorted(np.unique(pred).tolist())}")

        mask = seg_gt >= 0
        acc = (pred[mask] == seg_gt[mask]).mean()
        print(f"    Pixel accuracy: {acc:.3f}")

        out_bn = os.path.join(VIZ_DIR, "buildingnet", name)
        save_comparison(coord, color, pred, seg_gt,
                        f"{out_bn}_comparison.png",
                        elev=25, azim=60, n_pts=args.points)
        save_gt_pred_comparison(coord, color, pred, seg_gt,
                                 f"{out_bn}_gt_pred.png",
                                 elev=25, azim=60, n_pts=args.points)
        save_closeup_panels(coord, color, pred, seg_gt, name, "BuildingNet",
                            os.path.join(VIZ_DIR, "buildingnet"),
                            n_closeups=args.n_closeups, n_pts=args.points)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n[4/4] Done!")
    print(f"  Output: {VIZ_DIR}")
    print(f"  S3DIS images:     {os.path.join(VIZ_DIR, 's3dis')}")
    print(f"  BuildingNet images: {os.path.join(VIZ_DIR, 'buildingnet')}")


if __name__ == "__main__":
    main()
