_base_ = './fcn_vit16_clip_skip_480x480_40k_pascal_context_59.py'
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6
)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001, 
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'norm': dict(decay_mult=0.)}
    )
)
# data = dict(
#     samples_per_gpu=2,
# )