import os
import sys
from accelerate.logging import get_logger
import pandas as pd
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from accelerate.utils import ProjectConfiguration
from diffusers import DDIMScheduler, AutoencoderKL
import time
import accelerate
from accelerate import Accelerator, accelerator
from lib.Diffusion_major import DiffusionTrainer
from lib.EfficientFormer import efficientformerv2
from lib.SMT import smt_t
from lib.SaliencyNet import EEEAC2, EfficientNetB0
from lib.Unetb import FGDiff
from tqdm.auto import tqdm
sys.path.append("../")
from utils.seed_all import seed_all

logger = get_logger(__name__, log_level="INFO")
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run salient Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--SD_path",
        type=str,
        default='./ckpt',
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./ckpt/model.pth",
        help="path for unet",
    )
    parser.add_argument(
        "--input_rgb_path",
        type=str,
        default='./FGDiff/datasets/ORSSD/test/Image',
        help="Path to the input image.",
    )
    parser.add_argument(
        "--input_gt_path",
        type=str,
        default='./datasets/ORSSD/test/GT',
        help="Path to the input modality x.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output_dir',
        help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=3,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=384,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 384.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output mask at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=25,
        help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    args = parser.parse_args()
    checkpoint_path = args.pretrained_model_path
    input_image_path = args.input_rgb_path
    input_gt_path = args.input_gt_path
    output_dir = os.path.join(args.output_dir, str(int(time.time())))
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    seed = args.seed
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = 1
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        seed = int(time.time())
    seed_all(seed)

    # Output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------Data----------------------------
    logging.info("Inference Image Path from {}".format(input_image_path))

    # -------------------- Model --------------------

    unet = FGDiff(dims=[64, 128, 256, 512], dim_input=1, embedding_dim=256, dim_output=1)
    backbone = smt_t()
    fp_net = EEEAC2(train_enc=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    backbone.load_state_dict(checkpoint['backbone'], strict=True)
    unet.load_state_dict(checkpoint['unet'], strict=True)
    fp_net.load_state_dict(checkpoint['fp_net'], strict=True)

    backbone.to(device, torch.float32)
    unet.to(device, torch.float32)
    fp_net.to(device, torch.float32)

    denoise_steps = 3
    ensemble_size = 1
    match_input_res = True
    batch_size = 1


    from utils.metrics import Smeasure, MAE, Emeasure, Fmeasure

    smeasure = Smeasure()
    mae = MAE()
    emeasure = Emeasure()
    fmeasure = Fmeasure()
    pipeline = DiffusionTrainer(
        unet=unet,
        backbone=backbone,
        fp_net=fp_net,
        sampling_timesteps=denoise_steps,
        ensemble_size=ensemble_size,
        processing_res=384,
        match_input_res=match_input_res,
        batch_size=batch_size,
        training_target="x0",
    )

    logging.info("loading pipeline whole successfully.")
    name = []
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        unet.eval()
        backbone.eval()
        fp_net.eval()
        for img_name in tqdm(os.listdir(input_image_path)):

            input_image_pil = Image.open(os.path.join(input_image_path, img_name))
            salient_pred, salient_uncert = pipeline.infer(input_image_pil)

            pred_save_path = os.path.join(output_dir, img_name.split('.jpg')[0] + '.png')
            if os.path.exists(pred_save_path):
                logging.warning(f"Existing file: '{pred_save_path}' will be overwritten")
            salient_pred = np.expand_dims(salient_pred, 2)
            salient_pred = np.repeat(salient_pred, 3, 2)
            salient_pred = (salient_pred * 255).astype(np.uint8)

            salient_pred = Image.fromarray(salient_pred)
            salient_pred.save(pred_save_path)

            if input_gt_path:
                input_gt_pil = os.path.join(input_gt_path, img_name.split('.')[0] + '.png')
                input_gt_np = np.array(Image.open(input_gt_pil).convert('L')).astype(np.float32) / 255.
                salient_pred_np = np.array(Image.open(pred_save_path).convert('L')).astype(np.float32) / 255.
                input_gt_np = (input_gt_np > 0.5).astype('float')

                smeasure.step(salient_pred_np, input_gt_np)
                mae.step(salient_pred_np, input_gt_np)
                emeasure.step(salient_pred_np, input_gt_np)
                fmeasure.step(salient_pred_np, input_gt_np)
                name.append(img_name)
        del pipeline
        torch.cuda.empty_cache()

        sm_results = smeasure.get_results()
        mae_results = mae.get_results()
        em_results = emeasure.get_results()
        fm_results = fmeasure.get_results()

        mean_em = np.mean(em_results['em']['curve'])
        max_em = np.max(em_results['em']['curve'])
        adp_em = em_results['em']['adp']
        adp_fm = fm_results['fm']['adp']
        mean_fm = np.mean(fm_results['fm']['curve'])
        max_fm = np.max(fm_results['fm']['curve'])

        logger.info(f"Smeasure: {sm_results['sm']:.4f}")
        logger.info(f"MAE: {mae_results['mae']:.4f}")
        logger.info(f"meanEm: {mean_em:.4f}")
        logger.info(f"maxEm: {max_em:.4f}")
        logger.info(f"adpEm: {adp_em:.4f}")
        logger.info(f"Adp-Fmeasure: {adp_fm:.4f}")
        logger.info(f"mean-Fmeasure: {mean_fm:.4f}")
        logger.info(f"max-Fmeasure: {max_fm:.4f}")

        sms = smeasure.sms
        maes = mae.maes
        adp_ems = emeasure.adaptive_ems
        adp_fms = fmeasure.adaptive_fms
        # 保存结果
        log_data = {'names': np.array(name).flatten(),
                    'Smeasures': np.array(sms).round(4).flatten(),
                    'MAEs': np.array(maes).round(4).flatten(),
                    'Adp_emeasures': np.array(adp_ems).round(4).flatten(),
                    'Adp_fmeasures': np.array(adp_fms).round(4).flatten(),
                    }
        df = pd.DataFrame(log_data)
        path_excel = os.path.join(output_dir, 'single_metrics.csv')
        with open(path_excel, mode='w', encoding='utf-8') as f:
            df.to_csv(path_excel, index=True)

        log_data_mean = {
            'Smeasure': np.array(sm_results['sm']).round(4).flatten(),
            'max_fmeasure': np.array(max_fm).round(4).flatten(),
            'mean_fmeasure': np.array(mean_fm).round(4).flatten(),
            'Adp_fmeasure': np.array(adp_fm).round(4).flatten(),
            'max_emeasure': np.array(max_em).round(4).flatten(),
            'mean_emeasure': np.array(mean_em).round(4).flatten(),
            'Adp_emeasure': np.array(adp_em).round(4).flatten(),
            'MAEs': np.array(mae_results['mae']).round(4).flatten(),
            }
        df_mean = pd.DataFrame(log_data_mean)
        path_excel = os.path.join(output_dir, 'mean_metrics.csv')
        with open(path_excel, mode='w', encoding='utf-8') as f:
            df_mean.to_csv(path_excel, index=True)
