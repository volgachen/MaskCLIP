_base_ = './maskclip_vit16_p10_480x480_40k_pascal_context_59.py'
model = dict(
    decode_head=dict(
        text_embeddings_path=None,
    ),
)