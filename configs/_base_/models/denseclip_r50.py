# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/RN50_clip_visual.pth',
    backbone=dict(
        type='ResNetClip',
        depth=50,
        norm_cfg=norm_cfg,
        contract_dilation=True),
    decode_head=dict(
        type='DenseClipHead',
        in_channels=2048,
        channels=0,
        num_classes=0,
        dropout_ratio=0,
        in_index=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        ),
        init_cfg=dict()
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
