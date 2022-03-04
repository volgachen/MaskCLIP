_base_ = [
    '../_base_/models/denseclip_vit16.py', '../_base_/datasets/coco-stuff164k.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=171,
        text_categories=171, 
        text_channels=512, 
        text_embeddings_path='pretrain/stuff_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth',
        # num_vote=1,
        # vote_thresh=0.5,
        # cls_thresh=0.5,
    ),
    # backbone=dict(output_cls_token=True),
)