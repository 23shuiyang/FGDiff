import copy
import logging
import os
import utils
from models.swa_utils import AveragedModel
from argparse import ArgumentParser
import torch
import torch.nn as nn
from models.carbon import CarbonAI
from models.get_datasets import get_datasets
from models.get_models import get_model_performance
from models.get_models import get_models
from train_model import train_pkd, train_ps_kd, train_baseline, train_self_kd, train_ema_kd, train_dda_skd, train_pkd_skd
from utils import create_dir, Log_r, model_load_state_dict, val, fix_seed, validate
import time
import sys
import warnings

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--dataset_dir", type=str, default="data/")
parser.add_argument('--input_size_h', default=384, type=int)
parser.add_argument('--input_size_w', default=384, type=int)
parser.add_argument('--no_workers', default=0, type=int)
parser.add_argument('--no_epochs', default=15, type=int)

parser.add_argument('--log_interval', default=20, type=int)
parser.add_argument('--lr_sched', default=True, type=bool)
parser.add_argument('--model_val_path', default="model.pt", type=str)
parser.add_argument('--model_salicon_path', default="efb0_salicon_baseline.pt", type=str)
parser.add_argument('--log_dir', type=str, default="logging")
parser.add_argument('--decoder', type=str, default="ours")
parser.add_argument('--is_f', type=bool, default=False)

parser.add_argument('--alpha', type=float, default=0.999)
parser.add_argument('--k', type=float, default=10)
parser.add_argument('--a', type=float, default=0.4)
parser.add_argument('--f0', type=float, default=0.2)
parser.add_argument('--f1', type=float, default=0.2)
parser.add_argument('--f2', type=float, default=0.2)
parser.add_argument('--f3', type=float, default=0.2)
parser.add_argument('--f4', type=float, default=0.2)

parser.add_argument('--kldiv', default=True, type=bool)
parser.add_argument('--cc', default=True, type=bool)
parser.add_argument('--nss', default=True, type=bool)
parser.add_argument('--sim', default=False, type=bool)
parser.add_argument('--l1', default=False, type=bool)
parser.add_argument('--auc', default=False, type=bool)
parser.add_argument('--cs', default=True, type=bool)

parser.add_argument('--kldiv_coeff', default=1.0, type=float)
parser.add_argument('--cc_coeff', default=1.0, type=float)
parser.add_argument('--sim_coeff', default=1.0, type=float)
parser.add_argument('--nss_coeff', default=1.0, type=float)
parser.add_argument('--l1_coeff', default=1.0, type=float)
parser.add_argument('--auc_coeff', default=1.0, type=float)
parser.add_argument('--cs_coeff', default=1.0, type=float)

parser.add_argument('--amp', action='store_true', default=True)
parser.add_argument('--dataset', default="salicon", type=str)

parser.add_argument('--student', default="efb0", type=str)
parser.add_argument('--teacher', default="efb4", type=str)

parser.add_argument('--readout', default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--mode', default="baseline", type=str)
parser.add_argument('--seed', default=3407, type=int)
args = parser.parse_args()
fix_seed(args.seed)


csv_log = Log_r()
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
args.save = '{}/{}-{}-{}-{}'.format(args.log_dir, args.dataset, args.student, args.mode, time.strftime("%m-%d-%H-%M"))
create_dir(args.save, scripts_to_save=None)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.info("Hyperparameter: %s" % args)
logging.info('-'*80)
codecarbon = CarbonAI(country_iso_code="THA", args=args)
codecarbon.start()



if args.dataset != "salicon":
    args.output_size = (384, 384)

# 导入学生模型和教师模型
student, teacher = get_models(args)

torch.multiprocessing.freeze_support()


logging.info("train datasets： %s" % args.dataset)


train_loader, val_loader = get_datasets(args)

if args.dataset != "salicon":
    args.output_size = (384, 384)
    utils.model_load_state_dict(args, student, teacher, args.model_salicon_path)

#teacher.load_state_dict(torch.load(args.model_salicon_path)["student"], strict=True)
macs_t, params_t, macs_s, params_s = get_model_performance(args, teacher, student)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    student = nn.DataParallel(student)
    if args.mode == "pkd" or args.mode == 'pkd+skd':
        teacher = nn.DataParallel(teacher)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.mode == "pkd" or args.mode == 'ps-kd' or args.mode == 'pkd+skd':
    student.to(device)
    teacher.to(device)
else:
    student.to(device)
    teacher = None


logging.info(device)

if args.mode == "ema-kd" or args.mode == "ddp-kd" or args.mode == "pkd+skd":
    ema_model = torch.optim.swa_utils.AveragedModel(model=student,
                                                    use_buffers=True,
                                                    device=device,
                                                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.alpha))
else:
    ema_model = None

if args.mode == "self-kd" or args.mode == "ddp-kd" or args.mode == "pkd+skd":
    swa_model = AveragedModel(model=student, use_buffers=True, device=device)
else:
    swa_model = None

if args.mode == "pkd":
    params_group = [
        {"params": list(filter(lambda p: p.requires_grad, teacher.parameters())), "lr": args.learning_rate * 0.1},
        {"params": list(filter(lambda p: p.requires_grad, student.parameters())), "lr": args.learning_rate},
    ]
    optimizer = torch.optim.Adam(params_group)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.no_epochs))
else:
    params_group = [
        {"params": list(filter(lambda p: p.requires_grad, student.parameters())), "lr": args.learning_rate},
    ]
    optimizer = torch.optim.Adam(params_group)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.no_epochs))

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

start = time.time()

for epoch in range(0, args.no_epochs):
    lr = optimizer.param_groups[0]['lr']
    logging.info(f"Epoch {epoch}: Learning rate : {lr}")
    if args.mode == "baseline":
        loss = train_baseline(student, optimizer, train_loader, epoch, device, args, scaler)

    elif args.mode == "pkd":
        loss = train_pkd(student, optimizer, train_loader, epoch, device, args, scaler, teacher)

    elif args.mode == "ps-kd":
        loss = train_ps_kd(student, optimizer, train_loader, epoch, device, args, scaler, teacher)
        teacher = copy.deepcopy(student).to(device)

    elif args.mode == "self-kd":
        loss = train_self_kd(student, optimizer, train_loader, epoch, device, args, scaler, swa_model)

    elif args.mode == "ema-kd":
        loss = train_ema_kd(student, optimizer, train_loader, epoch, device, args, scaler, ema_model)
        if epoch == 0:
            ema_model.update_parameters(student)

    elif args.mode == "dda-skd":
        loss = train_dda_skd(student, optimizer, train_loader, epoch, device, args, scaler, ema_model, swa_model)
        if epoch == 0:
            ema_model.update_parameters(student)
    elif args.mode == "pkd+skd":
        loss = train_pkd_skd(student, optimizer, train_loader, epoch, device, args, scaler, ema_model, swa_model, teacher)
        if epoch == 0:
            ema_model.update_parameters(student)
    if args.lr_sched:
        scheduler.step()

    with torch.no_grad():
        _, cc_loss = utils.validate(student, val_loader, epoch, device, csv_log, lr, args)
        if epoch == 0:
            best_loss = cc_loss
        if best_loss >= cc_loss:
            best_loss = cc_loss
            if args.mode == "self-kd" or args.mode == "ddp-kd" or args.mode == "pkd+skd":
                swa_model.update_parameters(student)
            utils.save_state_dict(student, teacher, args)
            logging.info('[{:2d},  save, {}]'.format(epoch, args.model_val_path))


end = time.time()
csv_log.done(os.path.join(args.save, 'val.csv'))
logging.info("Time = %.2f Sec", (end - start))
logging.info("Emission = %.4f kgCO2" % codecarbon.emissions)
logging.info("Power = %.2f W" % codecarbon.power)
logging.info("macs-teacher = %s" % macs_t)
logging.info("macs-student = %s" % macs_s)
logging.info("macs-teacher = %s" % params_t)
logging.info("macs-student = %s" % params_s)

with torch.no_grad():
    model_load_state_dict(args, student, teacher, os.path.join(args.save, args.model_val_path))
    val(args, student, val_loader, device, codecarbon)