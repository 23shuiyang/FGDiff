import logging
import sys
import cv2
import numpy as np
from PIL import Image
from numpy import random
from ptflops import get_model_complexity_info
from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader.saliency_prediction_loader import get_datasets
from lib.SaliencyNet import EEEAC2
import time
import warnings
import os
import torchvision.transforms as transforms
def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def model_load_state_dict(model, path_state_dict):
    model.load_state_dict(torch.load(path_state_dict)['fp_net'], strict=True)
    print("loaded pre-trained model")
def test(model, device, args):
    model.eval()
    for img_name in tqdm(os.listdir(args.dataset_path)):
        input_image_pil = Image.open(os.path.join(args.dataset_path, img_name))
        input_image = input_image_pil.resize(args.output_size)
        # Convert the image to RGB and Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((args.input_size_h, args.input_size_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        input_image = input_image.convert("RGB")
        img = transform(input_image)
        img = img.to(torch.float32)
        img = img.to(device)
        img = img.unsqueeze(0)
        pred_map = model(img)
        pred_map = pred_map.squeeze(1)
        pred_save_path = os.path.join(args.save, img_name.split('.jpg')[0] + '.png')
        pred_map = normalize_map(pred_map)
        pred_map = pred_map.squeeze(0)
        pred_map = pred_map.detach().cpu().numpy().astype(np.float32)
        salient_pred = np.expand_dims(pred_map, 2)
        salient_pred = np.repeat(salient_pred, 3, 2)
        salient_pred = (salient_pred * 255).astype(np.uint8)
        salient_pred = Image.fromarray(salient_pred)
        salient_pred.save(pred_save_path)

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="./datasets/ORSI-4199/test/Image")
parser.add_argument('--input_size_h', default=384, type=int)
parser.add_argument('--input_size_w', default=384, type=int)
parser.add_argument('--model_path', default="/ckpt/model.pth", type=str)
parser.add_argument('--log_dir', type=str, default="./output_dir/")
parser.add_argument('--output_size', default=(384, 384))
parser.add_argument('--seed', default=25, type=int)
args = parser.parse_args()
fix_seed(args.seed)

# 创建记录文件
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
args.save = '{}/{}'.format(args.log_dir, time.strftime("%m-%d-%H-%M"))
if not os.path.exists(args.save):
    os.mkdir(args.save)
# 导入模型
model = EEEAC2(train_enc=True)
torch.multiprocessing.freeze_support()
args.output_size = (384, 384)
model_load_state_dict(model, args.model_path)
# 将模型导入GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
test(model, device, args)