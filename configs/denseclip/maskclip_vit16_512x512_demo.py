_base_ = [
    '../_base_/models/denseclip_vit16.py', '../_base_/datasets/maskclip_demo.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='../pretrain/ViT16_clip_visual.pth',
    # backbone=dict(return_qkv=False),
    decode_head=dict(
        num_classes=0,
        text_categories=0, 
        text_channels=512,
        text_embeddings_path='../pretrain/demo_ViT16_clip_text.pth',
        visual_projs_path='../pretrain/ViT16_clip_weights.pth',
        # num_vote=1,
        # vote_thresh=1.0,
        # cls_thresh=0.5,
        # bg_thresh = 0.5,
    ),
)