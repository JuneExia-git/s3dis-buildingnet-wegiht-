#!/bin/bash
# Run segmentation visualization
# Usage: bash run_viz.sh

cd /home/yang/PointCloud_Datasets
conda activate pointcept

python /mnt/c/Yang/Pointcept-main/Pointcept-main/exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet/code/viz_semseg_comparison.py \
    --s3dis_n 2 \
    --buildingnet_n 2 \
    --n_closeups 5 \
    --points 20000
