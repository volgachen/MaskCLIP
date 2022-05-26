_base_ = './fcn_vit16_p0_480x480_40k_pascal_context_59.py'
model = dict(
    pretrained='pretrain/vit-base-p16.pth',
)