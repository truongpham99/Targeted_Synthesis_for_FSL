import sys

import torch
from open_clip import image_transform
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import CLIPModel
import matplotlib.pyplot as plt
from SeedOptimizationStableDiffusion import SeedOptimizationStableDiffusion
import torch.nn.functional as F
import numpy as np
from encode_images import vae_encode, clip_encode
import os

CUDA = "cuda"

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def clip_img_transform(feature_extractor):
    image_mean = feature_extractor.image_mean
    image_std = feature_extractor.image_std
    preprocess_val = image_transform(
        feature_extractor.size["shortest_edge"],
        is_train=False,
        mean=image_mean,
        std=image_std
    )
    return preprocess_val


clip = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
clip.vision_model.to(CUDA)
clip.visual_projection.to(CUDA)
# freeze all models parameters
clip.eval()
freeze_params(clip.parameters())

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                            set_alpha_to_one=False, steps_offset=1)

model_id = "stabilityai/stable-diffusion-2-1-base"
sd_model = SeedOptimizationStableDiffusion.from_pretrained(model_id,
                                                            scheduler=scheduler)
sd_model = sd_model.to(CUDA)

batch = 6
prompt = ["black horse"]

height = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
width = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
shape = (batch, sd_model.unet.in_channels, height // sd_model.vae_scale_factor, width // sd_model.vae_scale_factor)

init_seed = torch.randn(shape, device=sd_model.device).to(
                CUDA) * sd_model.scheduler.init_noise_sigma
_, _, imgs = sd_model.apply(prompt=prompt*batch,
                                    guidance_scale=7.5,
                                    img_seed=init_seed,
                                    run_raw_sd=True)
# for i, img in enumerate(image_pil):
#     img.save(f"./results/image{i}.JPEG")

from open_clip import image_transform

def clip_img_transform(feature_extractor):
    image_mean = feature_extractor.image_mean
    image_std = feature_extractor.image_std
    preprocess_val = image_transform(
        feature_extractor.size["shortest_edge"],
        is_train=False,
        mean=image_mean,
        std=image_std
    )
    return preprocess_val

def transform_sd_tensors(imgs, preprocess):
    imgs = F.interpolate(imgs, size=preprocess[0].size, mode="bicubic")
    imgs = preprocess[-1](imgs)
    return imgs

clip_transform = clip_img_transform(sd_model.feature_extractor)
image_preproceesed = transform_sd_tensors(imgs, clip_transform.transforms)
print(image_preproceesed.shape)
