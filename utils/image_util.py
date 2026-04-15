import math
import os

import cv2
import torch
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    resized_img = img.resize((max_edge_resolution, max_edge_resolution))
    return resized_img


def pyramid_noise_like(noise, device, iterations=6, discount=0.3):
    b, c, w, h = noise.shape
    u = torch.nn.Upsample(size=(w, h), mode='bilinear').to(device)
    for i in range(iterations):
        r = random.random()*2+2 # Rather than always going 2x,
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(device)) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    c = s_map.size(1)
    w = s_map.size(2)
    h = s_map.size(3)

    min_s_map = torch.min(s_map.view(batch_size, 1, -1), 2)[0].view(batch_size, 1, 1, 1).expand(batch_size, c, w, h)
    max_s_map = torch.max(s_map.view(batch_size, 1, -1), 2)[0].view(batch_size, 1, 1, 1).expand(batch_size, c, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map

