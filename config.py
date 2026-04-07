weight = None
resume = False
evaluate = True
test_only = False
seed = 2614372
save_path = 'exp/yang/semseg-pt-v3m2-joint-s3dis-scannet-buildingnet'
num_worker = 8
batch_size = 4
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 400
eval_epoch = 20
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = True
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.2
param_dicts = [dict(keyword='block', lr=0.0003)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='MultiDatasetTrainer')
test = dict(type='SemSegTester', verbose=True)
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.003, 0.0003],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
UNIFIED_NAMES = [
    'wall', 'floor_ground', 'ceiling', 'roof', 'beam', 'column', 'window',
    'door_entrance', 'stairs', 'railing_fence', 'balcony_corridor_canopy',
    'molding_parapet_buttress', 'tower_chimney_dome', 'furniture_object',
    'vegetation_vehicle', 'garage', 'roof_detail', 'pool', 'other'
]
CORE_CLASS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
CLASS_WEIGHTS = [
    1.0, 1.0, 1.0, 1.2, 2.5, 2.5, 1.5, 1.5, 2.2, 2.0, 2.0, 2.0, 1.8, 1.0, 1.0,
    1.2, 1.5, 1.2, 0.5
]
S3DIS_REMAP = [2, 1, 0, 4, 5, 6, 7, 13, 13, 13, 13, 13, 18]
BUILD_REMAP = [
    0, 6, 14, 3, 14, 7, 12, 13, 1, 4, 8, 5, 9, 12, 12, 15, 9, 16, 10, 10, 11,
    7, 10, 17, 16, 11, 11, 6, 16, 18, -1, -1
]
model = dict(
    type='DefaultSegmentorV2',
    num_classes=19,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m2',
        in_channels=6,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False),
    criteria=[
        dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            ignore_index=-1,
            weight=[
                1.0, 1.0, 1.0, 1.2, 2.5, 2.5, 1.5, 1.5, 2.2, 2.0, 2.0, 2.0,
                1.8, 1.0, 1.0, 1.2, 1.5, 1.2, 0.5
            ]),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
data = dict(
    num_classes=19,
    ignore_index=-1,
    names=[
        'wall', 'floor_ground', 'ceiling', 'roof', 'beam', 'column', 'window',
        'door_entrance', 'stairs', 'railing_fence', 'balcony_corridor_canopy',
        'molding_parapet_buttress', 'tower_chimney_dome', 'furniture_object',
        'vegetation_vehicle', 'garage', 'roof_detail', 'pool', 'other'
    ],
    core_class_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='S3DISDataset',
                split=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5',
                       'Area_6'),
                data_root='/home/yang/PointCloud_Datasets/S3DIS_processed',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RandomDropout',
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2),
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='x',
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='y',
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='ChromaticAutoContrast', p=0.2,
                        blend_factor=None),
                    dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
                    dict(type='ChromaticJitter', p=0.95, std=0.05),
                    dict(
                        type='GridSample',
                        grid_size=0.02,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', sample_rate=0.6, mode='random'),
                    dict(type='SphereCrop', point_max=102400, mode='random'),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='NormalizeColor'),
                    dict(type='NormalizeCoord'),
                    dict(
                        type='RemapSegment',
                        remap=[2, 1, 0, 4, 5, 6, 7, 13, 13, 13, 13, 13, 18]),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment'),
                        feat_keys=('coord', 'color'))
                ],
                test_mode=False,
                loop=4),
            dict(
                type='BuildingNetDataset',
                split='train',
                data_root=
                '/home/yang/PointCloud_Datasets/BuildingNet_processed',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RandomDropout',
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2),
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='x',
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='y',
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='ChromaticAutoContrast', p=0.2,
                        blend_factor=None),
                    dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
                    dict(type='ChromaticJitter', p=0.95, std=0.05),
                    dict(
                        type='GridSample',
                        grid_size=0.05,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', point_max=102400, mode='random'),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='NormalizeNormal'),
                    dict(type='NormalizeColor'),
                    dict(
                        type='RemapSegment',
                        remap=[
                            0, 6, 14, 3, 14, 7, 12, 13, 1, 4, 8, 5, 9, 12, 12,
                            15, 9, 16, 10, 10, 11, 7, 10, 17, 16, 11, 11, 6,
                            16, 18, -1, -1
                        ]),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment'),
                        feat_keys=('color', 'normal'))
                ],
                test_mode=False,
                loop=1)
        ],
        loop=20),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='S3DISDataset',
                split='Area_5',
                data_root='/home/yang/PointCloud_Datasets/S3DIS_processed',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RemapSegment',
                        remap=[2, 1, 0, 4, 5, 6, 7, 13, 13, 13, 13, 13, 18]),
                    dict(
                        type='Copy', keys_dict=dict(segment='origin_segment')),
                    dict(
                        type='GridSample',
                        grid_size=0.02,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True,
                        return_inverse=True),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='NormalizeColor'),
                    dict(type='NormalizeCoord'),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment',
                              'origin_segment', 'inverse'),
                        feat_keys=('coord', 'color'))
                ],
                test_mode=False),
            dict(
                type='BuildingNetDataset',
                split='val',
                data_root=
                '/home/yang/PointCloud_Datasets/BuildingNet_processed',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RemapSegment',
                        remap=[
                            0, 6, 14, 3, 14, 7, 12, 13, 1, 4, 8, 5, 9, 12, 12,
                            15, 9, 16, 10, 10, 11, 7, 10, 17, 16, 11, 11, 6,
                            16, 18, -1, -1
                        ]),
                    dict(
                        type='Copy', keys_dict=dict(segment='origin_segment')),
                    dict(
                        type='GridSample',
                        grid_size=0.05,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True,
                        return_inverse=True),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='NormalizeNormal'),
                    dict(type='NormalizeColor'),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment',
                              'origin_segment', 'inverse'),
                        feat_keys=('color', 'normal'))
                ],
                test_mode=False)
        ]),
    test=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root='/home/yang/PointCloud_Datasets/S3DIS_processed',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'color'))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [1, 1]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1, 1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }]])))
