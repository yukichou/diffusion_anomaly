import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import os
import json
import numpy as np
import random

import torch
from PIL import Image, ImageDraw

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


def random_mask(im_shape, ratio=1.0, mask_full_image=False):

    width, height = im_shape
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    w_mask = random.randint(0, int(width * ratio))
    h_mask = random.randint(0, int(height * ratio))

    if mask_full_image:
        w_mask = int(width * ratio)
        h_mask = int(height * ratio)

    lim_w = width - w_mask // 2
    lim_h = height - h_mask // 2
    cx = random.randint(w_mask // 2, lim_w) if lim_w >= (w_mask // 2) else width // 2
    cy = random.randint(h_mask // 2, lim_h) if lim_h >= (h_mask // 2) else height // 2

    draw_type = random.randint(0, 1) if not mask_full_image else 0

    x0 = cx - w_mask // 2
    y0 = cy - h_mask // 2
    x1 = cx + w_mask // 2
    y1 = cy + h_mask // 2

    if draw_type == 0 or mask_full_image:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    else:
        draw.ellipse([x0, y0, x1, y1], fill=255)

    return mask  # PIL Image('L'), 0~255


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


class DDIMScheduler(SchedulerMixin, ConfigMixin):


    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        skip_prk_steps: bool = True,
    ):
        self.skip_prk_steps = skip_prk_steps

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )

    @classmethod
    def from_pretrained(cls, scheduler_folder):
        config_path = os.path.join(scheduler_folder, "scheduler_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(scheduler_folder, "scheduler", "scheduler_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Could not find scheduler_config.json in {scheduler_folder} or its 'scheduler' subfolder"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def _get_variance_before(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance_before = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance_before

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps}."
            )
        self.num_inference_steps = num_inference_steps


        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"{self.config.timestep_spacing} is not supported.")
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        anomaly_strength: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        mask_for_anomaly: Optional[torch.Tensor] = None,
        use_random_mask: bool = False,
        random_mask_ratio: float = 1.0,
        random_mask_full_image: bool = False
    ) -> Union[DDIMSchedulerOutput, Tuple[torch.Tensor]]:

        if self.num_inference_steps is None:
            raise ValueError("Need to call set_timesteps(...) after creating the scheduler")

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # 1) pred_original_sample
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(f"Unknown prediction_type: {self.config.prediction_type}")

        # 2) threshold or clip
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 3) direction
        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

        variance_before = self._get_variance_before(timestep, prev_timestep)
        std_dev_t = eta * variance_before.sqrt()

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

        # 4) standard ddim noise
        if eta > 0:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            prev_sample = prev_sample + std_dev_t * variance_noise

        # 5) anomaly_strength + mask_for_anomaly
        if anomaly_strength > 0:
            if variance_noise is None:
                noise2 = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            else:
                noise2 = variance_noise
            add_scale = variance_before.sqrt() * math.sqrt(anomaly_strength)

            final_mask = None


            if (mask_for_anomaly is not None) and (mask_for_anomaly.shape[0] == sample.shape[0]):
                if mask_for_anomaly.ndim == 4 and mask_for_anomaly.shape[1] == 1:
                    # repeat channel
                    mask_for_anomaly = mask_for_anomaly.repeat(1, sample.shape[1], 1, 1)
                final_mask = mask_for_anomaly

            if use_random_mask:
                bsz, ch, h, w = sample.shape
                random_mask_tensors = []
                for ib in range(bsz):
                    rm_pil = random_mask(
                        (w, h),
                        ratio=random_mask_ratio,
                        mask_full_image=random_mask_full_image
                    )
                    rm_arr = np.array(rm_pil, dtype=np.uint8)
                    rm_tensor = torch.from_numpy(rm_arr).float() / 255.0  # shape(H,W), 0/1
                    rm_tensor = rm_tensor.unsqueeze(0)  # (1,H,W)
                    if ch > 1:
                        rm_tensor = rm_tensor.repeat(ch,1,1)
                    random_mask_tensors.append(rm_tensor)

                random_mask_batch = torch.stack(random_mask_tensors, dim=0).to(sample.device)  # (B,C,H,W)
                if final_mask is not None:
                    final_mask = final_mask * random_mask_batch
                else:
                    final_mask = random_mask_batch

            if final_mask is not None:
                prev_sample += add_scale * noise2 * final_mask
            else:
                prev_sample += add_scale * noise2

        if not return_dict:
            return (prev_sample,)
        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps
