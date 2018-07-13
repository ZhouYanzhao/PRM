from typing import Union, Optional, List, Tuple

import cv2
import numpy as np
import torch.nn as nn
from torchvision import models
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import center_of_mass
from nest import register

from .models import FC_ResNet 
from .modules import PeakResponseMapping


@register
def fc_resnet50(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes)
    return model


@register
def peak_response_mapping(
    backbone: nn.Module,
    enable_peak_stimulation: bool = True,
    enable_peak_backprop: bool = True,
    win_size: int = 3,
    sub_pixel_locating_factor: int = 1,
    filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone, 
        enable_peak_stimulation = enable_peak_stimulation,
        enable_peak_backprop = enable_peak_backprop, 
        win_size = win_size, 
        sub_pixel_locating_factor = sub_pixel_locating_factor, 
        filter_type = filter_type)
    return model


@register
def prm_visualize(
    instance_list: List[dict], 
    class_names: Optional[List[str]]=None,
    font_scale: Union[int, float] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction visualization.
    """

    # helper functions
    def rgb2hsv(r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        s = 0 if mx == 0 else df / mx
        v = mx
        return h / 360.0, s, v

    def color_palette(N):
        cmap = np.zeros((N, 3))
        for i in range(0, N):
            uid = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (((uid & (1 << 0)) != 0) << 7 - j))
                g = np.bitwise_or(g, (((uid & (1 << 1)) != 0) << 7 - j))
                b = np.bitwise_or(b, (((uid & (1 << 2)) != 0) << 7 - j))
                uid = (uid >> 3)
            cmap[i, 0] = min(r + 86, 255)
            cmap[i, 1] = min(g + 86, 255)
            cmap[i, 2] = b
        cmap = cmap.astype(np.float32) / 255
        return cmap

    if len(instance_list) > 0:
        palette = color_palette(len(instance_list) + 1)
        height, width = instance_list[0]['mask'].shape[0], instance_list[0]['mask'].shape[1]
        instance_mask = np.zeros((height, width, 3), dtype=np.float32)
        peak_response_map = np.zeros((height, width, 3), dtype=np.float32)
        for idx, pred in enumerate(instance_list):
            category, mask, prm = pred['category'], pred['mask'], pred['prm']
            # instance masks
            instance_mask[mask, 0] = palette[idx + 1][0]
            instance_mask[mask, 1] = palette[idx + 1][1]
            instance_mask[mask, 2] = palette[idx + 1][2]
            if class_names is not None:
                y, x = center_of_mass(mask)
                y, x = int(y), int(x)
                text = class_names[category]
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                cv2.putText(
                    instance_mask,
                    text,
                    (x - text_size[0] // 2, y),
                    font_face,
                    font_scale,
                    (1., 1., 1.),
                    thickness)
            # peak response map
            peak_response = (prm - prm.min()) / (prm.max() - prm.min())
            mask = peak_response > 0.01
            h, s, _ = rgb2hsv(palette[idx + 1][0], palette[idx + 1][1], palette[idx + 1][2])
            peak_response_map[mask, 0] = h
            peak_response_map[mask, 1] = s
            peak_response_map[mask, 2] = np.power(peak_response[mask], 0.5)

        peak_response_map =  hsv_to_rgb(peak_response_map)
        return instance_mask, peak_response_map
