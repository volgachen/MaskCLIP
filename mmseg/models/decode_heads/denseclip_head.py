# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from ..builder import HEADS
from .decode_head import BaseDecodeHead

import math

@HEADS.register_module()
class DenseClipHead(BaseDecodeHead):

    def __init__(self, text_categories, text_channels, text_embeddings_path,
                    visual_projs_path, vit=False, conf_thresh=0., key_voting=True, **kwargs):
        super(DenseClipHead, self).__init__(**kwargs)

        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.visual_projs_path = visual_projs_path

        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))
        
        self.vit = vit
        if vit:
            self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.k_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.c_proj = nn.Conv2d(self.in_channels, text_channels, 1)

        self.conf_thresh = conf_thresh
        self.key_voting = key_voting

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
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded proj weights from {self.visual_projs_path}', logger=get_root_logger())
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        q, k = None, None
        if self.vit:
            if isinstance(x, list) and len(x) == 3:
                x, q, k = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            feat = self.proj(x)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            q = torch.flatten(q, start_dim=2).transpose(-2, -1)
            k = torch.flatten(k, start_dim=2).transpose(-2, -1)
            v = self.v_proj(x)
            feat = self.c_proj(v)
        output = self.cls_seg(feat)
        if k is not None and self.key_voting:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            k = k / math.sqrt(k.shape[2])
            attn = torch.bmm(k, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            output = torch.bmm(attn, output)
            output = output.transpose(-2, -1).view(N, C, H, W)
        return output

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        bg_output = None
        if self.text_categories > self.num_classes:
            bg_output, _ = torch.max(output[:, self.num_classes:], dim=1, keepdim=True)
        elif self.conf_thresh > 0:
            N, C, H, W = output.shape
            output = F.softmax(output*100, dim=1)
            bg_output = output.new_full((N, 1, H, W), self.conf_thresh)
        if bg_output is not None:
            output = torch.cat([bg_output, output[:, :self.num_classes]], dim=1)

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        raise RuntimeError('DenseCLIP is not trainable. Try DenseCLIP+ instead.')