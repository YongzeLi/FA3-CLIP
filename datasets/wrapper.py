"""
Function: DatasetWrapper with transfomrs in Unified Attack Detection
Author: Haocheng Yuan
Date: 2024/3/25

Base on:
Function: DatasetWrapper with transforms
Author: AJ
Date: 2022/7/7
"""

import os, random, torch, cv2
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from util.utils_FAS import check_if_exist
import torch.nn as nn

def get_params(resize, crop_size, degrees=180):
    w, h = resize, resize
    x = random.randint(0, np.maximum(0, w - crop_size))
    y = random.randint(0, np.maximum(0, h - crop_size))
    flip = random.random() > 0.5
    rotate = random.random() > 0.5
    angle = random.uniform(-degrees, degrees)
    ColorJitter = random.random() > 0.5
    brightness, contrast, saturation, hue = \
        random.random()/2.0, random.random()/2.0, random.random()/2.0, random.random()/2.0
    return {'crop_pos': (x, y), 'flip': flip, 'rotate': rotate, 'angle': angle, 'ColorJitter':ColorJitter,
            'brightness': brightness, 'contrast': contrast, 'saturation': saturation, 'hue': hue}
def _crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img
def _flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
def _rotate(img, angle):
    return img.rotate(angle)
def _get_ColorJitter(brightness, contrast, saturation, hue):
    """Get a randomized transform to be applied on image.
    Returns:
        Transform which randomly adjusts brightness, contrast and saturation in a random order.
    """
    transforms_list = []
    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        transforms_list.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))
    random.shuffle(transforms_list)
    return transforms_list

def get_transform(preprocess, params, load_size=300, crop_size=256, depth_size=32, cheek_size=16, is_depth=False):
    transform_list = []
    cheek_list = []
    if 'resize' in preprocess:transform_list.append(transforms.Resize([load_size, load_size]))
    if 'crop' in preprocess:
        if params is None:transform_list.append(transforms.RandomCrop(crop_size))
        else:transform_list.append(transforms.Lambda(lambda img: _crop(img, params['crop_pos'], crop_size)))
    else:
        osize = [crop_size, crop_size]
        transform_list.append(transforms.Resize(osize))
    if params is not None:
        if ('flip' in preprocess) and (params['flip']):transform_list.append(transforms.Lambda(lambda img: _flip(img)))
        if ('rotate' in preprocess) and (params['rotate']):
            transform_list.append(transforms.Lambda(lambda img: _rotate(img, params['angle'])))
        if ('ColorJitter' in preprocess) and (params['ColorJitter']):
            ColorJitter = transforms.ColorJitter(brightness=params['brightness'], contrast=params['contrast'], saturation=params['saturation'], hue=params['hue'])
            transform_list += _get_ColorJitter(ColorJitter.brightness, ColorJitter.contrast, ColorJitter.saturation, ColorJitter.hue)
    cheek_list += transform_list
    cheek_list.append(transforms.CenterCrop(cheek_size))
    cheek_list.append(transforms.Resize([crop_size, crop_size]))
    cheek_list += [transforms.ToTensor()]
    cheek_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if is_depth:
        transform_list.append(transforms.Resize([depth_size, depth_size]))
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
    return transforms.Compose(transform_list), transforms.Compose(cheek_list)


class FASD_RGB(data.Dataset):
    def __init__(self, net_name, data_source, image_size, depth_size, preprocess):
        super(FASD_RGB, self).__init__()
        self.data_source = data_source
        self.load_size = 300
        self.rotate = 180
        self.image_size = image_size
        self.depth_size = depth_size
        self.net_name = net_name
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data_source)

    def get_data(self, path, input_transform):
        img = Image.open(path).convert('RGB')
        img_r = input_transform(img)
        return img_r

    def __getitem__(self, index):
        item = self.data_source[index]
        x_path, x_label, y_path, y_label = item.impath_x, 0, item.impath_y, 1
        t_params = get_params(self.load_size, crop_size=self.image_size, degrees=self.rotate)
        input_transform, _ = get_transform(self.preprocess, params=t_params, crop_size=self.image_size, is_depth=False)
        x_r = self.get_data(x_path, input_transform)
        y_r = self.get_data(y_path, input_transform)
        return {'X_R': x_r, 'X_T': 'fake', 'X_L': x_label,
                'Y_R': y_r, 'Y_T': 'live', 'Y_L': y_label}

class FASD_RGB_VAL(data.Dataset):
    def __init__(self, data_source, image_size, preprocess):
        super(FASD_RGB_VAL, self).__init__()
        self.data_source = data_source
        self.load_size = 300
        self.rotate = 180
        self.image_size = image_size
        self.preprocess = preprocess
    def __len__(self):
        return len(self.data_source)
    def __getitem__(self, index):
        item = self.data_source[index]
        path, label = item.impath_x, item.label
        ### apply the same transform to both input and depth
        t_params = get_params(self.load_size, crop_size=self.image_size, degrees=self.rotate)
        input_transform, _ = get_transform(self.preprocess, params=t_params, crop_size=self.image_size, is_depth=False)
        frame = input_transform(Image.open(path).convert('RGB'))
        return {'frame': frame, 'label': label, 'text': 'none'}
