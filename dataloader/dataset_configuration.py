import os
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append("..")

from dataloader import transforms
from dataloader.salient_object_loader import SalientObjectDataset


def prepare_dataset(train_img_path=None,
                    train_gt_path=None,
                    batch_size=1,
                    datathread=4,
                    processing_res=384,
                    logger=None):

    train_dataset = SalientObjectDataset(train_img_path=train_img_path,
                                         train_gt_path=train_gt_path,
                                         processing_res=processing_res)
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=datathread, pin_memory=True)

    return train_loader


def gt_normalization(gt):
    min_value = torch.min(gt)
    max_value = torch.max(gt)
    normalized_gt = ((gt - min_value)/(max_value - min_value + 1e-5) - 0.5) * 2
    return normalized_gt
