_base_ = './maskclip_r50_512x512_coco-stuff164k.py'

model = dict(
    pretrained='pretrain/RN50x16_clip_visual.pth',
    backbone=dict(
        stem_channels=96,
        base_channels=96,
        depth='50x16'
    ),
    decode_head=dict(
        in_channels=3072,
        text_channels=768, 
        text_embeddings_path='pretrain/stuff_RN50x16_clip_text.pth',
        visual_projs_path='pretrain/RN50x16_clip_weights.pth',
        # num_vote=1,
        # vote_thresh=1.0,
        # cls_thresh=0.5,
        # attn_pooling=True,
        # num_heads=48
    )
)