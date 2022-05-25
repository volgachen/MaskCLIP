_base_ = './maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py'

model = dict(
    decode_head=dict(
        # num_vote=1,
        # vote_thresh=1.0,
        cls_thresh=0.5,
        # conf_thresh = 0.1,
    ),
)