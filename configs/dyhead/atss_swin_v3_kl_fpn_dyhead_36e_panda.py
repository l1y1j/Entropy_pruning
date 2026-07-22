_base_ = [
    '../_base_/datasets/panda_detection.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['sparse_former.models.backbones', 'sparse_former.models.detectors'],
    allow_failed_imports=False
)

# pretrained = '/data/linyujie/projects/Entropy_pruning/pretrained_model/swin_tiny_patch4_window7_224.pth'

work_dir = '/data/linyujie/projects/Entropy_pruning/outputs/v3_dyhead/run3'

find_unused_parameters = True
model = dict(
    type='ATSSWithGateLoss',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SwinTransformerV3',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained),

        # ===== Entropy Pruning =====
        strategy='kl_inc',

        # KL strategy config
        stage_config={
            0: {'blocks': [0, 1], 'ratio': [0.7, 0.7]},
            1: {'blocks': [0, 1], 'ratio': [0.7, 0.7]},
            2: {'blocks': [0, 1, 2, 3, 4, 5], 'ratio': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]},
            3: {'blocks': [0, 1], 'ratio': [0.7, 0.7]},
        },

        # INC strategy config
        inc_stage_config={
            0: {'blocks': [1], 'inc_ratio': 0.7},
            1: {'blocks': [0, 1], 'inc_ratio': [0.7, 0.7]},
            2: {'blocks': [0, 1, 2, 3, 4, 5], 'inc_ratio': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]},
            3: {'blocks': [0, 1], 'inc_ratio': [0.7, 0.7]},
        },

        # Learnable gate config
        use_learnable_gate=True,
        temperature=0.5,
        lambda_kl=2.0,
        lambda_inc=2.0,
    ),
    neck=[
        dict(
            type='FPN',
            in_channels=[192, 384, 768],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=1,
        in_channels=256,
        pred_kernel_size=1,  # follow DyHead official implementation
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),  # follow DyHead official implementation
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline),
    num_workers=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    # paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[33],
        gamma=0.1)
]

auto_scale_lr = dict(base_batch_size=16)
