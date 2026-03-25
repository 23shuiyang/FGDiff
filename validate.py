import os
from argparse import ArgumentParser
import torch
from models.carbon import CarbonAI
from models.get_datasets import get_datasets
from models.get_models import get_models, get_model_performance
from utils import model_load_state_dict, val, fix_seed
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
parser.add_argument('--model_val_path', default=r".\test\model.pt", type=str)
parser.add_argument('--model_salicon_path', default="model.pt", type=str)
parser.add_argument('--log_dir', type=str, default="logging")
parser.add_argument('--decoder', type=str, default="ours")
parser.add_argument('--is_f', type=bool, default=False)

parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--k', type=float, default=10)

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

parser.add_argument('--mode', default="dda-skd", type=str)
parser.add_argument('--seed', default=100, type=int)
args = parser.parse_args()

args.save = r'.\logging'
fix_seed(args.seed)

if args.dataset != "salicon":
    args.output_size = (384, 384)

student, teacher = get_models(args)
macs_t, params_t, macs_s, params_s = get_model_performance(args, teacher, student)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codecarbon = CarbonAI(country_iso_code="THA", args=args)
codecarbon.start()
if args.mode == "pkd" or args.mode == 'ps-kd':
    student.to(device)
    teacher.to(device)
else:
    student.to(device)
    teacher = None
torch.multiprocessing.freeze_support()

train_loader, val_loader = get_datasets(args)

with torch.no_grad():
    model_load_state_dict(args, student, teacher, os.path.join(args.save, args.model_val_path))
    if args.mode == "pkd":
        # print("Teacher:")
        # _ = validate(teacher, val_loader, device)
        print("Student:")
        val(args, student, val_loader, device, codecarbon)
    else:
        val(args, student, val_loader, device, codecarbon)