# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WebImageDataset(CustomDataset):

    from tools.denseclip_utils.prompt_engineering import bg_classes
    CLASSES = ['obj1', 'obj2'] + bg_classes
    PALETTE = [[255, 0, 0], [0, 0, 255]] + [[0, 0, 0]] * len(bg_classes)

    # CLASSES = ('baseball player', 'basketball player',
    #            'soccer player', 'football player',
    #            'person', 'background', 'wall', 'building',
    #            'sky', 'grass', 'tree', 'ground', 'floor',
    #            'baseball court', 'basketball court', 'soccer court', 'football court')
    # PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]] + [[0, 0, 0]] * (len(CLASSES)-4)

    # CLASSES = ('Bugatti Veyron', 'Cadillac DeVille',
    #            'Porsche 718 Cayman', 'Lamborghini Gallardo'
    #            'road', 'sidewalk', 'building', 'wall', 
    #            'fence', 'pole', 'traffic light', 'traffic sign', 
    #            'vegetation', 'terrain', 'sky', 'person', 'rider', 
    #            'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background')
    # PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]] + [[0, 0, 0]] * (len(CLASSES)-4)

    # CLASSES = ('blurry car', 'sharp car', 'road',
    #            'sidewalk', 'building', 'wall',
    #            'fence', 'pole', 'traffic light',
    #            'traffic sign', 'vegetation', 'terrain',
    #            'sky', 'person', 'rider',
    #            'truck', 'bus', 'train',
    #            'motorcycle', 'bicycle')
    # PALETTE = [[255, 0, 0], [0, 0, 255]] + [[0, 0, 0]] * (len(CLASSES)-2)


    def __init__(self, **kwargs):
        super(WebImageDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)