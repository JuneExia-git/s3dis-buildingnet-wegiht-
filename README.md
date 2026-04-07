# s3dis-buildingnet-wegiht-（固定预训练权重跨数据集分割实验）

## 实验概述

**模型**: PT-v3m2 + DefaultSegmentorV2 (19 类统一语义分割)  
**任务**: 跨数据集语义分割 — S3DIS（室内）和 BuildingNet（建筑）联合训练  
**策略**: 固定预训练编码器权重（freeze_encoder=False 实际仅微调），冻结 backbone

## 数据集

| 数据集 | 用途 | 描述 |
|--------|------|------|
| S3DIS | 训练 + 验证 | 6 Area 训练，Area_5 验证 |
| BuildingNet | 训练 + 验证 | train/val/test 拆分 |

**统一类别（19 类）**: wall, floor_ground, ceiling, roof, beam, column, window, door_entrance, stairs, railing_fence, balcony_corridor_canopy, molding_parapet_buttress, tower_chimney_dome, furniture_object, vegetation_vehicle, garage, roof_detail, pool, other

## 训练配置

- **Epoch**: 400
- **Batch Size**: 4
- **学习率**: OneCycleLR, max_lr=0.003
- **优化器**: AdamW, weight_decay=0.05
- **Loss**: CrossEntropy + Lovasz
- **AMP**: 开启 (float16)

## 模型权重

- **Best Model**: `model/model_best.pth`（核心 mIoU 最优）
- **Last Model**: `model/model_last.pth`

## 可视化结果

- `viz/training_curves_miou_acc.png` — 训练过程 mIoU / mAcc 曲线
- `viz/training_curves_core_classes.png` — 10 核心类别 IoU 曲线
- `viz/training_curves_core_classes_bar.png` — 最终 vs 最佳对比柱状图
- `viz/s3dis/` — S3DIS 测试集可视化（gt_pred / comparison / closeup）
- `viz/buildingnet/` — BuildingNet 测试集可视化

## 核心脚本

- `code/viz_semseg_comparison.py` — 语义分割可视化（GT vs Pred）
- `code/plot_training_curves.py` — 从 train.log 绘制训练曲线
- `run_viz.sh` — 可视化运行脚本
- `config.py` — 完整训练配置

## 依赖

- Pointcept
- PyTorch >= 1.9
- numpy, matplotlib
- trimesh, open3d（可选，用于更高级的 3D 渲染）

---
*实验时间: 2026-03*
