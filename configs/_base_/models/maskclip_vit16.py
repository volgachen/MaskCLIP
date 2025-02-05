# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MCVisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_bias=False,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm = True,
        final_norm=True,
        return_qkv=True,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False
    ),
    decode_head=dict(
        type='MaskClipHead',
        vit=True,
        in_channels=768,
        channels=0,
        num_classes=20,
        text_categories=20,
        text_channels=512,
        dropout_ratio=0,
        in_index=-1,
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
)  # yapf: disable
