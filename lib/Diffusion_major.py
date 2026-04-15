import numpy as np
import copy
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from lib.EfficientFormer import efficientformerv2
from lib.SaliencyNet import EEEAC2, EfficientNetB0
from lib.Unetb import FGDiff
from lib.simple_diffusion import unnormalize_to_zero_to_one, default, logsnr_schedule_cosine, logsnr_schedule_shifted, \
    right_pad_dims_to
from model.diffusion_decoder.diffusion_utils import get_beta_schedule, to_torch
from utils.ensemble import ensemble_masks
from utils.image_util import resize_max_res, normalize_map, vis
import torch
from torch import sqrt
from torch.special import expm1

from tqdm import tqdm
from einops import repeat

class DiffusionTrainer(object):
    def __init__(self,
                 unet: FGDiff,
                 backbone: efficientformerv2,
                 fp_net: EfficientNetB0,
                 sampling_timesteps=10,
                 ensemble_size=1,
                 processing_res=384,
                 match_input_res=True,
                 batch_size=1,
                 training_target="x0",
                 ):
        self.log_snr = logsnr_schedule_cosine
        self.log_snr = logsnr_schedule_shifted(self.log_snr, processing_res, noise_d=64)
        self.sampling_timesteps = sampling_timesteps
        self.unet = unet
        self.sample_type = "ddim"
        self.backbone = backbone
        self.fp_net = fp_net
        self.ensemble_size = ensemble_size
        self.processing_res = processing_res
        self.match_input_res = match_input_res
        self.batch_size = batch_size
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        self.training_target = training_target
        assert self.training_target in ["v", "x0", "eps"]

        self.model_var_type = "fixedlarge"
        betas = get_beta_schedule(
            beta_schedule="cosine",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = to_torch(betas).to(self.device)
        alphas = 1.0 - betas
        alphas_hat = alphas.cumprod(dim=0)
        alphas_hat_prev = torch.cat([torch.ones(1).to(device), alphas_hat[:-1]], dim=0)
        self.betas = betas
        self.alphas_hat = alphas_hat
        self.alphas_hat_prev = alphas_hat_prev
        self.sqrt_alphas_hat = torch.sqrt(alphas_hat)
        self.sqrt_one_minus_alphas_hat = torch.sqrt(1.0 - alphas_hat)
        self.log_one_minus_alphas_hat = torch.log(1.0 - alphas_hat)
        self.sqrt_recip_alphas_hat = torch.sqrt(1.0 / alphas_hat)
        self.sqrt_recipm1_alphas_hat = torch.sqrt(1.0 / alphas_hat - 1)

        posterior_variance = betas * (1.0 - alphas_hat_prev) / (1.0 - alphas_hat)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.maximum(posterior_variance, torch.tensor(1e-20))
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_hat) / (1.0 - alphas_hat)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_hat_prev) * torch.sqrt(alphas) / (1.0 - alphas_hat)
        )
        self.num_timesteps = betas.shape[0]

    def infer(self, input_image: Image):
        input_size = input_image.size

        # --------------- Image Processing ------------------------
        # Resize image
        input_image = resize_max_res(
            input_image, max_edge_resolution=self.processing_res
        )
        # Convert the image to RGB and Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((self.processing_res, self.processing_res)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        input_image = input_image.convert("RGB")
        rgb = transform(input_image)
        rgb = rgb.to(torch.float32)
        rgb = rgb.to(self.device)

        # ----------------- predicting salient -----------------
        duplicated_rgb = torch.stack([rgb] * self.ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=self.batch_size, shuffle=False)

        loader = zip(single_rgb_loader)
        # predict the salient
        salient_pred_ls = []
        # Remove progress bars to prevent multiple bars during validation
        for _, batched_image in enumerate(loader):
            salient_pred_raw = self.sample(rgb_in=batched_image[0][0])
            salient_pred_ls.append(salient_pred_raw.detach().clone())

        salient_preds = torch.concat(salient_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if self.ensemble_size > 1:
            salient_pred, salient_uncert = ensemble_masks(salient_preds)
        else:
            salient_pred = salient_preds
            salient_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(salient_pred)
        max_d = torch.max(salient_pred)
        salient_pred = (salient_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        salient_pred = salient_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if self.match_input_res:
            pred_img = Image.fromarray(salient_pred)
            pred_img = pred_img.resize(input_size)
            salient_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        salient_pred = salient_pred.clip(0, 1)

        return salient_pred, salient_uncert

    @torch.no_grad()
    def sample(self, rgb_in, verbose=True):
        B, C, H, W = rgb_in.shape
        if self.sample_type == "ddim":
            x = self.p_sample_ddim(shape=(B, 1, H, W), rgb_in=rgb_in, verbose=verbose)
        elif self.sample_type == "ddpm":
            x = self.p_sample_ddpm(shape=(B, 1, H, W), rgb_in=rgb_in, verbose=verbose)
        torch.cuda.empty_cache()
        # clip prediction
        salient = x
        return salient


    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    @torch.no_grad()
    def p_sample_ddim(self, shape, rgb_in, verbose=True):
        """
        DDIM sampling with log-SNR scheduler (0→1).
        shape : (B, C, H, W)
        cond_img : backbone features list/tuple
        """
        img = torch.randn(shape, device=self.device)
        if self.fp_net is not None:
            sal_map = self.fp_net(rgb_in)
            sal_map = normalize_map(sal_map)
        else:
            sal_map=None
        conditioning_features = self.backbone.forward_encode(rgb_in, sal_map)

        steps = torch.linspace(1., 0., self.sampling_timesteps + 1, device=self.device)
        for i in range(self.sampling_timesteps):
            time = steps[i]
            time_next = steps[i + 1]
            img = self.p_sample_ddim_step(img, conditioning_features,
                                          time, time_next, sal_map)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample_ddim_step(self, x, cond, time, time_next, sal_map):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)

        squared_alpha = log_snr.sigmoid()
        squared_alpha_next = log_snr_next.sigmoid()
        alpha = squared_alpha.sqrt()
        alpha_next = squared_alpha_next.sqrt()
        sigma = (-log_snr).sigmoid().sqrt()

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        if sal_map is not None:
            x_s = torch.cat([x, sal_map], dim=1)
            pred = self.unet(cond, batch_log_snr, x_s)
        else:
            pred = self.unet(cond, batch_log_snr, x)

        # 反推 x₀
        if self.training_target == 'v':
            x0 = alpha * x - sigma * pred
        elif self.training_target == 'eps':
            x0 = (x - sigma * pred) / alpha
        elif self.training_target == 'x0':
            x0 = pred.tanh()
        else:
            raise ValueError(self.training_target)
        x0.clamp_(-1., 1.)

        # ---- 关键：t_next==0 时直接返回 x₀ ----
        if time_next <= 0.:
            return x0

        # 否则继续 DDIM 更新
        sigma_next = (-log_snr_next).sigmoid().sqrt()
        x_pred = alpha_next * x0 + sigma_next * ((x - alpha * x0) / sigma)
        return x_pred

    @torch.no_grad()
    def p_sample_ddpm(self, shape, cond_img, rgb_in, verbose=True):
        self.history = []

        img = torch.randn(shape, device=self.device)
        if self.fp_net is not None:
            sal_map = self.fp_net(rgb_in)
            sal_map = normalize_map(sal_map)
        else:
            sal_map=None
        conditioning_features = self.backbone(cond_img)

        steps = torch.linspace(1., 0., self.sampling_timesteps + 1, device=self.device)

        for i in tqdm(range(self.sampling_timesteps), desc='sampling loop time step', total=self.sampling_timesteps,
                      disable=not verbose, leave=True, position=0):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, conditioning_features, times, times_next, sal_map)


        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample(self, x, cond, time, time_next, sal_map):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, time=time,
                                                          time_next=time_next, sal_map=sal_map)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_sample_g(self, x, cond, time, time_next, sal_map):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, time=time,
                                                          time_next=time_next, sal_map=sal_map)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_mean_variance(self, x, cond, time, time_next, sal_map):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        if sal_map is not None:
            x_s = torch.cat([x, sal_map], dim=1)
            pred = self.unet(cond, batch_log_snr, x_s)
        else:
            pred = self.unet(cond, batch_log_snr, x)
        if self.training_target == 'v':
            x_start = alpha * x - sigma * pred

        elif self.training_target == 'eps':
            x_start = (x - sigma * pred) / alpha

        elif self.training_target == 'x0':
            # raise NotImplementedError
            # due to we don't know x is normalized or not
            x_start = pred.tanh()
            # x_start = x

        x_start.clamp_(-1., 1.)
        self.history.append(x_start)  # change to pred when generate cam

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance