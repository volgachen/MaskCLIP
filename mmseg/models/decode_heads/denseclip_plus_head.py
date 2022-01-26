# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from mmseg.ops import resize
from ..builder import HEADS, build_head, build_backbone
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class DenseClipPlusHead(BaseDecodeHead):

    def __init__(self, decode_module_cfg, text_categories, text_channels, 
                    text_embeddings_path, 
                    cls_bg=False, norm_feat=False, start_self_train=(-1, -1), start_clip_guided=(-1, -1),
                    unlabeled_cats=[], clip_unlabeled_cats=[], clip_cfg=None, clip_weights_path=None,
                    reset_counter=False, clip_channels=None,
                    **kwargs):
        super(DenseClipPlusHead, self).__init__(
            input_transform=decode_module_cfg.pop('input_transform'), **kwargs)
        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.norm_feat = norm_feat
        self.unlabeled_cats = torch.tensor(unlabeled_cats, device='cuda')
        self.clip_unlabeled_cats = torch.tensor(clip_unlabeled_cats, device='cuda')
        self.start_self_train = start_self_train
        self.start_clip_guided = start_clip_guided
        self.self_train = (start_self_train[0] >= 0)
        self.clip_guided = (start_clip_guided[0] >= 0)
        self.train_unlabeled = self.self_train or self.clip_guided
        self.register_buffer('_iter_counter', torch.tensor(0, device='cuda'))
        self.clip_weights_path = clip_weights_path
        self.cls_bg = cls_bg
        self.reset_counter = reset_counter
        if clip_channels is None:
            clip_channels = self.in_channels

        del self.conv_seg
        self.init_cfg = None

        decode_module_cfg.update(kwargs)
        self.build_decode_module(decode_module_cfg)

        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))

        if self.clip_guided:
            self.clip = build_backbone(clip_cfg)
            self.v_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.c_proj = nn.Conv2d(clip_channels, text_channels, 1)

        if cls_bg:
            self.bg_embeddings = nn.Parameter(torch.randn(1, text_channels))

    def init_weights(self, call_super=True):
        if call_super:
            super(DenseClipPlusHead, self).init_weights()
        self.load_text_embeddings()
        if self.clip_guided:
            self.load_clip_weights()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())

    def load_clip_weights(self):
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        self.clip.load_state_dict(loaded['clip'])
        for attr in ['v_proj', 'c_proj']:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                # Linear to Conv2d
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded clip weights from {self.clip_weights_path}', logger=get_root_logger())

    def _freeze(self):
        """Freeze params and norm stats."""
        super(DenseClipPlusHead, self)._freeze()
        # always freeze these modules
        if self.clip_guided:
            for i in [self.clip, self.v_proj, self.c_proj]:
                for m in i.modules():
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        # never freeze bg_classifier
        if self.cls_bg:
            self.bg_embeddings.requires_grad = True
    

    def build_decode_module(self, cfg):
        cfg['init_cfg'] = None
        cfg['in_channels'] = self.in_channels
        cfg['channels'] = self.channels
        self.decode_module = build_head(cfg)
        del self.decode_module.loss_decode
        del self.decode_module.conv_seg
        del self.decode_module.dropout

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        if self.norm_feat:
            feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        
        if self.cls_bg:
            bg_weight = self.bg_embeddings / self.bg_embeddings.norm(dim=-1, keepdim=True)
            bg = F.conv2d(feat, bg_weight[:, :, None, None])
            output = torch.cat([bg, output], dim=1)
        
        return output

    def forward(self, inputs):
        output = self.decode_module.forward_module(inputs)

        feat = output.detach()
        output = self.cls_seg(output)

        if self.reset_counter:
            self.reset_counter = False
            self._iter_counter *= 0

        self._iter_counter += 1
        if self.training:
            if self._iter_counter == self.start_self_train[0]:
                print_log('Start self training', logger=get_root_logger())
            if self._iter_counter == self.start_self_train[1]:
                print_log('Stop self training', logger=get_root_logger())
            if self._iter_counter == self.start_clip_guided[0]:
                print_log('Start clip guided training', logger=get_root_logger())
            if self._iter_counter == self.start_clip_guided[1]:
                print_log('Stop clip guided training', logger=get_root_logger())

        if self.train_unlabeled:
            return [output, feat]

        return [output]


    def assign_label(self, gt_semantic_seg, feat, norm=False, unlabeled_cats=None):
        if (gt_semantic_seg < 0).sum() == 0:
            return gt_semantic_seg, None

        feat = resize(
            input=feat,
            size=gt_semantic_seg.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if norm:
            feat = feat / feat.norm(dim=1, keepdim=True)

        feat = feat.permute(0, 2, 3, 1)
        gt_semantic_seg = gt_semantic_seg.squeeze(1)

        unlabeled_idx = (gt_semantic_seg < 0)
        # [candidates, channels]
        unlabeled_feat = feat[unlabeled_idx]
        if self.cls_bg:
            bg_embeddings = self.bg_embeddings / self.bg_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = torch.cat([bg_embeddings, self.text_embeddings], dim=0)
        else:
            text_embeddings = self.text_embeddings
        # [unlabeled_cats, text_channels]
        unlabeled_text = text_embeddings[unlabeled_cats]
        # [candidates, unlabeled_cats]
        match_matrix = unlabeled_feat @ unlabeled_text.t()

        gt_semantic_seg[unlabeled_idx] = unlabeled_cats[match_matrix.argmax(dim=1)]

        return gt_semantic_seg[:, None, :, :]

    def label_sanity_check(self, gt_semantic_seg):
        for i in self.unlabeled_cats:
            assert torch.all(gt_semantic_seg != i), f'Ground-truth leakage! {i}'
        for i in self.clip_unlabeled_cats:
            assert torch.all(gt_semantic_seg != i), f'Ground-truth leakage! {i}'

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.train_unlabeled:
            seg_logits, feat = self.forward(inputs)
            gt_self, gt_clip, gt_weight = None, None, None
            self.label_sanity_check(gt_semantic_seg)
            if not torch.all(gt_semantic_seg != -1):
                if self.self_train and self._iter_counter >= self.start_self_train[0] and \
                    (self._iter_counter <= self.start_self_train[1] or self.start_self_train[1] < 0):
                    with torch.no_grad():
                        gt = gt_semantic_seg.clone()
                        gt_self = self.assign_label(gt, feat,
                                    self.norm_feat, self.unlabeled_cats)
                        del gt
                if self.clip_guided and self._iter_counter >= self.start_clip_guided[0] and \
                    (self._iter_counter <= self.start_clip_guided[1] or self.start_clip_guided[1] < 0):
                    with torch.no_grad():
                        # clip cannot deal with background
                        gt = gt_semantic_seg.clone()
                        if gt_self is not None and self.cls_bg:
                            gt[gt_self == 0] = 0
                        feat = self.clip(img)[-1]
                        feat = self.c_proj(self.v_proj(feat))
                        gt_clip = self.assign_label(gt, feat,
                                    True, self.clip_unlabeled_cats)
                        del gt
                if gt_self is not None:
                    gt_semantic_seg = gt_self
                if gt_clip is not None:
                    # merge gt_self and gt_clip
                    if gt_self is not None:
                        for i in self.trust_clip_on:
                            idx = (gt_clip == i)
                            gt_semantic_seg[idx] = i
                    else:
                        gt_semantic_seg = gt_clip

                # ignore the unlabeled
                gt_semantic_seg[gt_semantic_seg<0] = 255
                
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            seg_logits = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]