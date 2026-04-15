import os
import sys
import math

import cv2
import torch
import shutil
import logging
import argparse
import datasets
import transformers
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from packaging import version
import torch.nn.functional as F
from transformers.utils import ContextManagers

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from lib.Diffusion_major import DiffusionTrainer
from lib.EfficientFormer import efficientformerv2
from lib.SMT import smt_t
from lib.SaliencyNet import EEEAC2, EfficientNetB0
from lib.Unetb import FGDiff
from lib.simple_diffusion import right_pad_dims_to
from utils.loss import Bce_iou_loss, structure_loss_with_ual

sys.path.append("..")
from dataloader.dataset_configuration import prepare_dataset, gt_normalization
from utils.image_util import normalize_map

logger = get_logger(__name__, log_level="INFO")
epoch_val_losses = []


def log_validation(unet, backbone, args, accelerator, epoch, fp_net=None):
    global epoch_val_losses
    denoise_steps = 3
    ensemble_size = 1
    match_input_res = True
    batch_size = 1
    logger.info("Running validation ... ")

    from utils.metrics import Smeasure, MAE, Emeasure, Fmeasure

    smeasure = Smeasure()
    mae = MAE()
    emeasure = Emeasure()
    fmeasure = Fmeasure()
    pipeline = DiffusionTrainer(
        unet=accelerator.unwrap_model(unet),
        backbone=accelerator.unwrap_model(backbone),
        fp_net=accelerator.unwrap_model(fp_net),
        sampling_timesteps=denoise_steps,
        ensemble_size=ensemble_size,
        processing_res=384,
        match_input_res=match_input_res,
        batch_size=batch_size,
        training_target="x0",
    )

    total_loss = 0.0
    val_bar = tqdm(
        range(0, len(os.listdir(args.val_img_path))),
        initial=0,
        desc="Validation Steps",
        disable=not accelerator.is_local_main_process,
        position=0,
        leave=True,
        ncols=100
    )
    unet.eval()
    backbone.eval()
    fp_net.eval()
    with torch.no_grad():
        for idx, img_name in enumerate(os.listdir(args.val_img_path)):
            input_image_path = args.val_img_path + img_name
            input_image_pil = Image.open(input_image_path)
            salient_pred, salient_uncert = pipeline.infer(input_image_pil)
            salient_pred = np.expand_dims(salient_pred, 2)
            salient_pred = np.repeat(salient_pred, 3, 2)
            salient_pred = Image.fromarray((salient_pred * 255).astype(np.uint8))
            pred_save_path = os.path.join(args.output_dir, 'val_image', img_name.split('.jpg')[0] + '.png')
            if os.path.exists(os.path.join(args.output_dir, 'val_image')) is not True:
                os.makedirs(os.path.join(args.output_dir, 'val_image'), exist_ok=True)
            salient_pred.save(pred_save_path)

            salient_pred_np = np.array(Image.open(pred_save_path).convert('L')).astype(np.float32) / 255.

            gt_path = args.val_gt_path + img_name.split('.')[0] + '.png'
            input_gt_np = np.array(Image.open(gt_path).convert('L')).astype(np.float32) / 255.
            input_gt_np = (input_gt_np > 0.5).astype('float')

            smeasure.step(salient_pred_np, input_gt_np)
            mae.step(salient_pred_np, input_gt_np)  # MAE使用浮点数GT
            emeasure.step(salient_pred_np, input_gt_np)
            fmeasure.step(salient_pred_np, input_gt_np)

            # 计算L1损失（保持原有的损失计算）
            signal_loss = F.l1_loss(torch.tensor(salient_pred_np), torch.tensor(input_gt_np))
            logs = {"signal_loss": signal_loss}
            val_bar.set_postfix(**logs)
            val_bar.update(1)

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

    logger.info(f"Epoch {epoch} - L1 Loss: {total_loss:.4f}")
    logger.info(f"Smeasure: {sm_results['sm']:.4f}")
    logger.info(f"MAE: {mae_results['mae']:.4f}")
    logger.info(f"meanEm: {mean_em:.4f}")
    logger.info(f"maxEm: {max_em:.4f}")
    logger.info(f"adpEm: {adp_em:.4f}")
    logger.info(f"Adp-Fmeasure: {adp_fm:.4f}")
    logger.info(f"mean-Fmeasure: {mean_fm:.4f}")
    logger.info(f"max-Fmeasure: {max_fm:.4f}")

    total_loss = mae_results['mae'] + (1-max_em) + (1-max_fm) + (1-sm_results['sm'])
    epoch_val_losses.append(total_loss)


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion-Based Image Generators for salient Detection")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='./ckpt',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_img_path",
        type=str,
        default='./datasets/ORSI-4199/train/Image/',
        help="train file listing the training image files",
    )
    parser.add_argument(
        "--train_gt_path",
        type=str,
        default='./datasets/ORSI-4199/train/GT/',
        help="train file listing the training gt files",
    )
    parser.add_argument(
        "--val_img_path",
        type=str,
        default=r'./datasets/ORSI-4199/test/Image/',
    )
    parser.add_argument(
        "--val_gt_path",
        type=str,
        default='./datasets/ORSI-4199/test/GT/',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=25, help="A seed for reproducible training.")
    parser.add_argument(
            "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=300)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--prediction_type",
        type=str,
        default="x0",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="salient-detection",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)"
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations)"
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=384,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 384.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    # set the warning levels
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.

    logger.info("loading the noise scheduler from {}".format(args.pretrained_model_name_or_path),
                main_process_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        unet = FGDiff(dims=[64, 128, 256, 512], dim_input=1, embedding_dim=256, dim_output=1)

    backbone = smt_t()
    backbone.load_state_dict(torch.load(args.pretrained_model_name_or_path + '/smt_tiny.pth')['model'], strict=False)
    fp_net = EEEAC2(train_enc=True)
    fp_net.load_state_dict(torch.load(args.pretrained_model_name_or_path + '/salicon_eeeac2.pt')['fp_net'], strict=True)

    # Freeze vae and set unet to trainable.
    noise_scheduler = DiffusionTrainer(
        unet=accelerator.unwrap_model(unet),
        backbone=accelerator.unwrap_model(backbone),
        fp_net=accelerator.unwrap_model(fp_net),
        training_target=args.prediction_type,
    )
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        [
            {
                "params": list(unet.parameters()) + list(backbone.parameters()),
                "lr": args.learning_rate,
            },
            {
                "params": fp_net.parameters(),
                "lr": args.learning_rate*0.01,
            },
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        train_loader = prepare_dataset(
            train_img_path=args.train_img_path,
            train_gt_path=args.train_gt_path,
            batch_size=args.train_batch_size,
            datathread=args.dataloader_num_workers,
            processing_res=args.processing_res,
            logger=logger)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(unet, optimizer, train_loader, lr_scheduler)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move vae and FGDiff_backbone to gpu and cast to weight_dtype
    backbone.to(accelerator.device, dtype=weight_dtype)
    fp_net.to(accelerator.device, dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Here is the DDP training: actually is 32
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    epoch_train_losses = []
    min_loss = float('inf')
    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        backbone.train()
        fp_net.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with (accelerator.accumulate(unet)):
                salient_image = batch['image']
                salient_gt = batch['gt']
                salient_seg = batch['seg']
                if salient_seg is not None:
                    mask_gt = gt_normalization(salient_seg)
                else:
                    mask_gt = gt_normalization(salient_gt)

                noise = torch.randn_like(mask_gt)
                bsz = mask_gt.shape[0]

                # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denoising.
                # Sample a random timestep for each image]
                timesteps = torch.zeros((bsz,), device=device).float().uniform_(0, 1)
                noisy_mask, log_snr = noise_scheduler.q_sample(x_start=mask_gt, times=timesteps, noise=noise)
                ########################################################################################################
                sal_map = fp_net.forward_encode(salient_image.to(weight_dtype))
                sal_map = normalize_map(sal_map)

                feats = backbone.forward_encode(salient_image.to(weight_dtype), sal_map.to(weight_dtype))

                if args.prediction_type == 'v':
                    padded_log_snr = right_pad_dims_to(noisy_mask, log_snr)
                    alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
                    target = alpha * noise - sigma * salient_gt

                elif args.prediction_type == 'eps':
                    target = noise

                elif args.prediction_type == 'x0':
                    target = salient_gt

                # predict the noise residual and compute the loss.
                unet_input = torch.cat([noisy_mask, sal_map], dim=1)
                pred = unet(feats, log_snr, unet_input)

                loss = Bce_iou_loss(pred=pred.float(), mask=target.float())
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                epoch_train_losses.append(loss.detach().item())
                # Back propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process:
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # saving the checkpoints
                if global_step % args.checkpointing_steps == 0 or (
                        loss.detach().item() < min_loss and loss.detach().item() < 0.01):
                    min_loss = loss.detach().item()
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            model_state = {
                'unet': accelerator.unwrap_model(unet).state_dict(),
                'backbone': accelerator.unwrap_model(backbone).state_dict(),
                'fp_net': accelerator.unwrap_model(fp_net).state_dict(),
            }
            torch.save(model_state, os.path.join(args.output_dir, f"model_{epoch}.pth"))

            model_files = [f for f in os.listdir(args.output_dir) if f.startswith("model_") and f.endswith(".pth")]
            model_files = [f for f in model_files if f != "model_best.pth"]  # 排除最佳模型文件
            if len(model_files) > 5:
                model_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                files_to_remove = model_files[:-5]
                for file_to_remove in files_to_remove:
                    os.remove(os.path.join(args.output_dir, file_to_remove))
                    logger.info(f"Removed old model file: {file_to_remove}")

            log_validation(
                unet=unet,
                backbone=backbone,
                args=args,
                accelerator=accelerator,
                epoch=epoch,
                fp_net=fp_net
            )
            last_value = epoch_val_losses[-1]
            is_min_value = last_value == min(epoch_val_losses)
            if is_min_value:
                best_model_state = {
                    'unet': accelerator.unwrap_model(unet).state_dict(),
                    'backbone': accelerator.unwrap_model(backbone).state_dict(),
                    'fp_net': accelerator.unwrap_model(fp_net).state_dict(),
                }
                torch.save(best_model_state, os.path.join(args.output_dir, "model_best.pth"))
                logger.info(f"Saved best model with validation loss: {last_value}")

                if os.path.exists(os.path.join(args.output_dir, "valcheckpoint")):
                    shutil.rmtree(os.path.join(args.output_dir, "valcheckpoint"))

            save_trainloss = os.path.join(args.output_dir, "train_losses.txt")
            with open(save_trainloss, 'w') as file:
                for loss in epoch_train_losses:
                    file.write(f"{loss}\n")
            print(f"Values overwritten to {save_trainloss}")

            save_valloss = os.path.join(args.output_dir, "val_losses.txt")
            with open(save_valloss, 'w') as file:
                for loss in epoch_val_losses:
                    file.write(f"{loss}\n")
            print(epoch_val_losses)
            print(f"Values overwritten to {save_valloss}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
