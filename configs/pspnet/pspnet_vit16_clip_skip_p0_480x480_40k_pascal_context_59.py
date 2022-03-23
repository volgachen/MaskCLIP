_base_ = './pspnet_vit16_p0_480x480_40k_pascal_context_59.py'
model = dict(
    pretrained='pretrain/ViT16_clip_visual.pth',
    backbone=dict(
        skip_last_attn=True
    )
)