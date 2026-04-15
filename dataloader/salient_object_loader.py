from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

import torch
from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataloader.boundary_modification import modify_boundary
from utils.image_util import read_text_lines, resize_max_res
import albumentations
import torchvision.transforms as transforms
def random_modified(gt, iou_max=1.0, iou_min=0.8):
    iou_target = np.random.rand()*(iou_max-iou_min)+iou_min
    seg = modify_boundary((np.array(gt) > 0.5).astype("uint8")*255, iou_target=iou_target)
    return seg

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
class SalientObjectDataset(Dataset):
    def __init__(self,
                 train_img_path,
                 train_gt_path,
                 processing_res=384):
        super(SalientObjectDataset, self).__init__()

        self.processing_res = processing_res
        self.samples = []
        lines_img = [train_img_path + f for f in os.listdir(train_img_path) if f.endswith('.jpg') or f.endswith('.png')]
        lines_gt = [train_gt_path + f for f in os.listdir(train_gt_path) if f.endswith('.png') or f.endswith('.jpg')]

        self.aug_transform = albumentations.Compose([
            albumentations.RandomScale(scale_limit=0.25, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=15, p=0.5),
            albumentations.RandomRotate90(p=0.5),
            # albumentations.ElasticTransform(p=0.5),
        ])
        self.img_transform = self.get_transform(None, None)
        self.gt_transform = transforms.Compose([
            transforms.Resize((processing_res, processing_res)),
            transforms.ToTensor()])

        for idx in range(len(lines_img)):
            sample = dict()
            sample['image'] = lines_img[idx]
            sample['gt'] = lines_gt[idx]
            self.samples.append(sample)

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.processing_res, self.processing_res)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        return transform

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        image = self.rgb_loader(sample_path['image'])
        gt = self.binary_loader(sample_path['gt'])
        image_size = image.size

        # data augumentation
        image, gt = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt)).values()
        image, gt = albumentations.PadIfNeeded(*image_size[::-1], border_mode=0)(image=image, mask=gt).values()
        image, gt = albumentations.RandomCrop(*image_size[::-1])(image=image, mask=gt).values()
        image = colorEnhance(Image.fromarray(image))
        seg = random_modified(gt)
        seg = self.gt_transform(Image.fromarray(seg))

        gt = Image.fromarray(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if len(gt.shape) == 2:
            tmp = gt[None, :, :]
            seg = seg[None, :, :]
        else:
            tmp = gt[0:1, :, :]
            seg = seg[0:1, :, :]
        sample['gt'] = tmp
        sample['image'] = image
        sample['seg'] = seg
        return sample

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.samples)
