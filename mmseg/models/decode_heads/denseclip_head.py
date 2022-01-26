# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class DenseClipHead(BaseDecodeHead):

    def __init__(self, text_categories, text_channels, text_embeddings_path,
                    visual_projs_path, **kwargs):
        super(DenseClipHead, self).__init__(**kwargs)

        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.visual_projs_path = visual_projs_path

        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))
        
        self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
        self.c_proj = nn.Conv2d(self.in_channels, text_channels, 1)
        
        self.load_text_embeddings()
        self.load_visual_projs()

    def init_weights(self):
        super(DenseClipHead, self).init_weights()
        self.load_text_embeddings()
        self.load_visual_projs()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())

    def load_visual_projs(self):
        loaded = torch.load(self.visual_projs_path, map_location='cuda')
        for attr in ['v_proj', 'c_proj']:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded proj weights from {self.visual_projs_path}', logger=get_root_logger())
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        v = self.v_proj(x)
        feat = self.c_proj(v)
        output = self.cls_seg(feat)
        return output

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        raise RuntimeError('DenseCLIP is not trainable. Try DenseCLIP+ instead.')