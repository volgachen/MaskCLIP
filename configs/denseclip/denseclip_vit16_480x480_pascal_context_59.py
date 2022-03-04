_base_ = [
    '../_base_/models/denseclip_vit16.py', '../_base_/datasets/pascal_context_59.py', 
    # '../_base_/models/denseclip_vit16.py', '../_base_/datasets/pascal_context.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(
        num_classes=59,
        text_categories=59, 
        text_channels=512, 
        text_embeddings_path='pretrain/context_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth',
        # num_vote=1,
        # vote_thresh=0.5,
        # topk_text=30,
        # cls_thresh=0.5,
        # bg_thresh = 0.1,
    ),
    # backbone=dict(output_cls_token=True),
)