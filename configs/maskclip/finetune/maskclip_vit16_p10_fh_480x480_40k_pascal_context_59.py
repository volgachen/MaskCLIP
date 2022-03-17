_base_ = './maskclip_vit16_p10_480x480_40k_pascal_context_59.py'
model = dict(
    decode_head=dict(
        freeze=True,
    ),
)
# data = dict(
#     samples_per_gpu=2,
# )