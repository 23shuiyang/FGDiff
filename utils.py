import torchvision
from tqdm import tqdm

import logging
import os
import shutil
import sys
import time
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
from scipy import io
from loss import *
import torch


class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def post_processing_A(x):
    #x = cv2.resize(x, img_size)
    x = (x - x.min()) / (x.max()-x.min())
    x = x * 255
    x = np.clip(np.round(x),0,255)
    return x

def post_processing_B(x):
    x = (x - x.min()) / (x.max()-x.min())
    x = x * 255
    x = x + 0.5
    x = np.clip(np.round(x),0,255)
    return x
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
def save_state_dict(student, teacher, args):
    if torch.cuda.device_count() > 1:
        if args.mode == "pkd":
            params = {
                'student': student.module.state_dict(),
                'teacher': teacher.module.state_dict()
            }
        else:
            params = {
                'student': student.module.state_dict()
            }
    else:
        if args.mode == "pkd":
            params = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict()
            }
        else:
            params = {
                'student': student.state_dict(),
            }
    torch.save(params, os.path.join(args.save, args.model_val_path))

def model_load_state_dict(args, student, teacher, path_state_dict):
    dicts = torch.load(path_state_dict)["student"]
    if args.mode == "pkd":
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        teacher.load_state_dict(torch.load(path_state_dict)["teacher"], strict=True)
        print("loaded pre-trained student and teacher")
    else:
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        print("loaded pre-trained student")


def loss_func(pred_map, gt, fixations, args):
    loss = torch.FloatTensor([0.0]).cuda()
    loss_nss = torch.FloatTensor([0.0]).cuda()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += (1 - args.cc_coeff * cc(pred_map, gt))
    if args.nss:
        nss_value = nss(pred_map, fixations)
        loss_nss += (1 - args.nss_coeff * (torch.exp(nss_value) / (1 + torch.exp(nss_value))))
    if args.sim:
        loss += (1 - args.sim_coeff * similarity(pred_map, gt))
    if args.auc:
        loss += (1 - args.auc_coeff * auc_judd(pred_map, fixations))
    if args.cs:
        loss += (1 - args.cs_coeff * cosine_similarity(pred_map, gt))
    return loss + loss_nss

def loss_func_features(f_s, f_t, args):
    loss = torch.FloatTensor([0.0]).cuda()
    loss += args.kldiv_coeff * kldiv(f_s, f_t)
    loss += (1 - args.cc_coeff * cc(f_s, f_t))
    loss += (1 - args.cs_coeff * cosine_similarity(f_s, f_t))
    return loss

class Log_r(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.results = []

    def update(self, epoch, lr, cc, kd, nss, sim, auc, total):
        self.results.append([epoch, lr, cc, kd, nss, sim, auc, total])

    def done(self, name):
        np.savetxt(name, np.array(self.results), delimiter=',', header="epoch,lr,cc,kd,nss,sim,auc,total", comments="",
                   fmt='%s')

def create_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def val(args, model, loader, device, codecarbon):
    args.num_visualize=1
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_loss = AverageMeter()
    start = time.time()
    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        if args.is_f:
            features, pred_map = model(img)
        else:
            pred_map = model(img)
        cc_loss.update(cc(pred_map, gt))
        kldiv_loss.update(kldiv(pred_map, gt))
        nss_loss.update(nss(pred_map, fixations))
        sim_loss.update(similarity(pred_map, gt))
        auc_loss.update(auc_judd(pred_map, fixations))
    end = time.time()

    print('CC : {:.4f}, KLDIV : {:.4f}, NSS : {:.4f}, SIM : {:.4f}, AUC : {:.4f}'.format(
        cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, auc_loss.avg))
    log_data = {"kd-mode": args.mode,
                "student": args.student,
                "teacher": args.teacher,
                "decoder": args.decoder,
                "CC": round((cc_loss.avg.cpu()).item(), 4),
                'KLDiv': round((kldiv_loss.avg.cpu()).item(), 4),
                'NSS': round((nss_loss.avg.cpu()).item(), 4),
                'SIM': round((sim_loss.avg.cpu()).item(), 4),
                'AUC': round(auc_loss.avg, 4),
                'time / munites': round((end - start) / 60, 4),
                'Emission / kgCO2': codecarbon.emissions,
                'Power / W': codecarbon.power,
                'file_name': args.save,
                }
    df = pd.DataFrame(log_data, index=[0])
    path_data = args.save + '/' + args.dataset
    #path_data = 'D:/project/PKD+SKD/test'
    if not os.path.exists(path_data):
        print("create a new folder: {}".format(path_data))
        os.makedirs(path_data)
    path_excel = os.path.join(path_data, 'val_avg.csv')
    with open(path_excel, mode='w', encoding='utf-8') as f:
        df.to_csv(path_excel, index=True)
    sys.stdout.flush()
def validate(model, loader, epoch, device, csv_log, lr, args):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_loss = AverageMeter()
    for (img, gt, fixations) in loader:
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        if args.is_f:
            _, pred_map = model(img)
        else:
            pred_map = model(img)
        cc_loss.update(cc(pred_map, gt))
        kldiv_loss.update(kldiv(pred_map, gt))
        nss_loss.update(nss(pred_map, fixations))
        sim_loss.update(similarity(pred_map, gt))
        auc_loss.update(auc_judd(pred_map, fixations))
    logging.info(
        '[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, AUC : {:.5f}  time:{:3f} minutes'.format(
            epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, auc_loss.avg, (time.time() - tic) / 60))
    sys.stdout.flush()

    nss_avg = ((torch.exp(nss_loss.avg) / (1 + torch.exp(nss_loss.avg))))
    metric_scores = torch.tensor([1 - cc_loss.avg, kldiv_loss.avg, 1 - nss_avg, 1 - sim_loss.avg, 1 - auc_loss.avg],
                                 dtype=torch.float32)
    if csv_log is not None:
        csv_log.update(epoch, lr, cc_loss.avg.item(), kldiv_loss.avg.item(), nss_loss.avg.item(), sim_loss.avg.item(),
                       auc_loss.avg.item(), torch.sum(metric_scores).item())
    return cc_loss.avg, torch.sum(metric_scores)