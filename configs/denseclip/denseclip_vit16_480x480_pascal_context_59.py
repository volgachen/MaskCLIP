_base_ = [
    '../_base_/models/denseclip_vit16.py', '../_base_/datasets/pascal_context_59.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        text_categories=59, 
        text_channels=512, 
        text_embeddings_path='pretrain/context_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth',
    ),
)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))
