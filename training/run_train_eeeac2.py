import logging
import sys
import cv2
import numpy as np
from numpy import random
from ptflops import get_model_complexity_info
from argparse import ArgumentParser
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from dataloader.saliency_prediction_loader import get_datasets
from lib.SaliencyNet import EEEAC2, EfficientNetB0
import time
import warnings
import os
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
def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map
def similarity(s_map, gt):
    ''' For single image metric
        Size of Image - WxH or 1xWxH
        gt is ground truth saliency map
    '''
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    s_map = normalize_map(s_map)
    gt = normalize_map(gt)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    return torch.mean(torch.sum(torch.min(s_map, gt), 1))
def auc_judd(saliencyMap, fixationMap, jitter=True, toPlot=False, normalize=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    #       ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve
    # If there are no fixations to predict, return NaN
    if saliencyMap.size() != fixationMap.size():
        saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
        saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(0)
        # saliencyMap = saliencyMap.cuda()
        # fixationMap = fixationMap.cuda()
    if len(saliencyMap.size())==3:
        saliencyMap = saliencyMap[0,:,:]
        fixationMap = fixationMap[0,:,:]
    saliencyMap = saliencyMap.cpu().detach().numpy()
    fixationMap = fixationMap.cpu().detach().numpy()
    if normalize:
        saliencyMap = normalize_map(saliencyMap)

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score
def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_gt.size() == gt.size()

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt / (s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))
def cc(s_map, gt):
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa*bb)))
def nss(s_map, gt):
    if s_map.size() != gt.size():
        s_map = s_map.cpu().squeeze(0).numpy()
        s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
        s_map = s_map.cuda()
        gt = gt.cuda()
    assert s_map.size()==gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.view(batch_size, -1), 1)
    return torch.mean(s_map / count)
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
def get_model_performance(args, model):
    macs, params = get_model_complexity_info(model, (3, args.input_size, args.input_size), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
    logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params
def model_load_state_dict(model, path_state_dict):
    model.load_state_dict(torch.load(path_state_dict), strict=True)
    print("loaded pre-trained model")
def loss_func(pred_map, gt, fixations, args):
    loss = torch.FloatTensor([0.0]).cuda()
    loss_nss = torch.FloatTensor([0.0]).cuda()
    loss += kldiv(pred_map, gt)
    loss += (1 - cc(pred_map, gt))
    nss_value = nss(pred_map, fixations)
    loss_nss += (1 - (torch.exp(nss_value) / (1 + torch.exp(nss_value))))
    return loss + loss_nss
def validate(model, loader, epoch, device, csv_log, lr):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_loss = AverageMeter()
    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        pred_map = model(img)
        pred_map = pred_map.squeeze(1)
        cc_loss.update(cc(pred_map, gt))
        kldiv_loss.update(kldiv(pred_map, gt))
        nss_loss.update(nss(pred_map, fixations))
        sim_loss.update(similarity(pred_map, gt))
        auc_loss.update(auc_judd(pred_map, fixations))
    print(
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

def train_baseline(student, optimizer, loader, epoch, device, args, scaler):
    student.train()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            pred_map_student = pred_map_student.squeeze(1)
            loss = loss_func(pred_map_student, gt, fixations, args)
        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        if idx % args.log_interval == (args.log_interval - 1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dataset_dir", type=str, default="./datasets/")
parser.add_argument('--input_size_h', default=384, type=int)
parser.add_argument('--input_size_w', default=384, type=int)
parser.add_argument('--no_epochs', default=10, type=int)
parser.add_argument('--log_interval', default=20, type=int)
parser.add_argument('--lr_sched', default=True, type=bool)
parser.add_argument('--model_val_path', default="model.pt", type=str)
parser.add_argument('--model_salicon_path', default="salicon_eeeac2.pt", type=str)
parser.add_argument('--log_dir', type=str, default="logging-FP")
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--dataset', default="SALICON", type=str)
parser.add_argument('--output_size', default=(384, 384))
parser.add_argument('--seed', default=25, type=int)
args = parser.parse_args()
fix_seed(args.seed)

# 创建记录文件
csv_log = Log_r()
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
args.save = '{}/{}-{}'.format(args.log_dir, args.dataset, time.strftime("%m-%d-%H-%M"))
if not os.path.exists(args.save):
    os.mkdir(args.save)

# 导入模型
model = EEEAC2(train_enc=True)
#model = EfficientNetB0(train_enc=True)
torch.multiprocessing.freeze_support()
# 记录
logging.info("train datasets： %s" % args.dataset)
# 导入训练集和测试集
train_loader, val_loader = get_datasets(args)

# 如果不是 SALICON 数据集，则导入预训练参数
if args.dataset != "SALICON":
    args.output_size = (384, 384)
    model_load_state_dict(model, args.model_salicon_path)

# 将模型导入GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 记录
logging.info(device)

params_group = [
    {"params": list(filter(lambda p: p.requires_grad, model.parameters())), "lr": args.learning_rate},
]
optimizer = torch.optim.Adam(params_group)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.no_epochs))

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

for epoch in range(0, args.no_epochs):
    lr = optimizer.param_groups[0]['lr']
    # 记录
    print(f"Epoch {epoch}: Learning rate : {lr}")
    loss = train_baseline(model, optimizer, train_loader, epoch, device, args, scaler)
    if args.lr_sched:
        scheduler.step()
    # 测试
    with torch.no_grad():
        cc_, cc_loss = validate(model, val_loader, epoch, device, csv_log, lr)
        if epoch == 0:
            best_loss = cc_loss
        if best_loss >= cc_loss:
            best_loss = cc_loss
            params = {
                'fp_net': model.state_dict(),
            }
            torch.save(params, os.path.join(args.save, args.model_val_path))
            print('[{:2d},  save best weight, {}]'.format(epoch, args.model_val_path))