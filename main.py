import sys

import torch
from open_clip import image_transform
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import CLIPModel
import matplotlib.pyplot as plt
from SeedOptimizationStableDiffusion import SeedOptimizationStableDiffusion, transform_sd_tensors
import torch.nn.functional as F
import numpy as np
from encode_images import vae_encode, clip_encode
import submod 
import os
# from nao.ddim_inversion import image_folder_to_ddim
# from nao.norm_aware_optimization import norm_aware_centroid_optimization

CUDA = "cuda"

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """
    """ Code from: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355"""

    if not isinstance(v0, np.ndarray):
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2).to(CUDA)

    return v2


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


def prepare_models():
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
    clip_transform = clip_img_transform(sd_model.feature_extractor)

    return sd_model, sd_model.vae, clip, clip_transform


def seedselect_loss(losses, latents, vae_centroid):
    weights = torch.tensor([0.05, 0.25, 0.7]).cuda().T

    semantic_loss = weights @ torch.stack(losses).squeeze(-1)
    appearance_loss = F.mse_loss(latents, vae_centroid, reduction="none").mean([1, 2, 3]).mean()
    return 10 * semantic_loss + appearance_loss
    
def gcmi_loss(images_timesteps, latents, clip_transform, vae_examplars, clip_examplars, alpha=10):
    weights = torch.tensor([0.05, 0.25, 0.7]).cuda().T

    clip_losses = []
    for images in images_timesteps:
        images_preproceesed = transform_sd_tensors(images.float(), clip_transform)
        clip_imgs_features = clip_model.get_image_features(images_preproceesed)
        clip_imgs_features_normed = clip_imgs_features / clip_imgs_features.norm(dim=-1, keepdim=True)
        clip_losses.append(submod.gcmi(clip_imgs_features_normed, clip_examplars))

    semantic_loss = weights @ torch.stack(clip_losses).squeeze(-1)
    appearance_loss = submod.gcmi(latents, vae_examplars)
    return alpha * semantic_loss + appearance_loss

def submod_optimize_seed(batch, init_seed, prompt, sd_model, clip_model, clip_transform, clip_examplars, vae_examplars,
                  submod_loss="gcmi", n_iters=30, guidance_scale=7.5, lr=0.05):
    img_seed = torch.nn.Parameter(init_seed.reshape((batch, 4, 64, 64)), requires_grad=True)

    optimizer = torch.optim.AdamW([img_seed], lr=lr)
    for i in tqdm(range(n_iters)):
        if i == 3:
            for g in optimizer.param_groups:
                g['lr'] = lr/10
        optimizer.zero_grad()
        latents, _, image, images_timesteps = sd_model.apply(prompt=[prompt]*batch,
                                                    img_seed=img_seed,
                                                    clip_model=clip_model,
                                                    guidance_scale=guidance_scale,
                                                    clip_transform=clip_transform,
                                                    clip_img_centroid=clip_centroid)
        if submod_loss == "gcmi":
            loss = gcmi_loss(images_timesteps, latents, clip_transform, vae_examplars, clip_examplars)
        loss.backward()
        optimizer.step()
        print("-"*30)

    os.makedirs("./results", exist_ok=True)
    os.makedirs(f"./results/submod_{submod_loss}/", exist_ok=True)
    os.makedirs(f"./results/submod_{submod_loss}/{prompt}", exist_ok=True)

    image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image_pils = sd_model.numpy_to_pil(image_numpy)
    for i in range(len(image_pil)):
        image_pils[i].save(f"./results/submod_{submod_loss}/{prompt}/image_{i}.JPEG")


def optimize_seed(init_seed, prompt, sd_model, clip_model, clip_transform, clip_centroid, vae_centroid, count,
                  n_iters=30, guidance_scale=7.5, lr=0.05):

    img_seed = torch.nn.Parameter(init_seed.reshape((1, 4, 64, 64)), requires_grad=True)

    optimizer = torch.optim.AdamW([img_seed], lr=lr)
    for i in tqdm(range(n_iters)):
        if i == 3:
            for g in optimizer.param_groups:
                g['lr'] = lr/10
        optimizer.zero_grad()
        latents, losses, image, _ = sd_model.apply(prompt=prompt,
                                                    img_seed=img_seed,
                                                    clip_model=clip_model,
                                                    guidance_scale=guidance_scale,
                                                    clip_transform=clip_transform,
                                                    clip_img_centroid=clip_centroid)

        loss = seedselect_loss(losses, latents, vae_centroid)
        loss.backward()
        optimizer.step()
        print("-"*30)
        # if i != 0 or show_first_image == True:
        #     image_pil.save(f"./results/{i}_image.JPEG")
        #     plt.imshow(image_pil)
        #     plt.axis("off")
        #     plt.tight_layout()
        #     plt.show()
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/ss/", exist_ok=True)
    os.makedirs(f"./results/ss/{prompt}", exist_ok=True)

    image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image_pil = sd_model.numpy_to_pil(image_numpy)
    image_pil[0].save(f"./results/ss/{prompt}/image_{count}.JPEG")

    return latents.detach()
    


if __name__ == "__main__":

    prompt = sys.argv[2]  # tail class
    print(prompt)

    img_folder = sys.argv[3] # path to folder with iamges of the rare concept

    sd_model, vae, clip_model, clip_transform = prepare_models()



    # Clip centroid from images
    clip_examplars = clip_encode(clip_model, clip_transform, img_folder).mean(dim=0)
    clip_centroid = clip_examplars / clip_examplars.norm(dim=-1, keepdim=True)

    # Create latent centroid from images
    vae_examplars = vae_encode(vae, img_folder)
    vae_centroid = vae_examplars.mean(dim=0).unsqueeze(0)

    height = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
    width = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
    shape = (1, sd_model.unet.in_channels, height // sd_model.vae_scale_factor, width // sd_model.vae_scale_factor)

    seed_everything(22)
    if sys.argv[1] == "StableDiffusion":
        # Run raw stable diffusion
        with torch.no_grad():
            init_seed = torch.randn(shape, device=sd_model.device, dtype=clip_centroid.dtype).to(
                CUDA) * sd_model.scheduler.init_noise_sigma
            _, _, image, _ = sd_model.apply(prompt=prompt,
                                             guidance_scale=7.5,
                                             img_seed=init_seed,
                                             run_raw_sd=True)
            # plt.imshow(image_pil)
            # plt.axis("off")
            # plt.tight_layout()
            # plt.show()
            image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            image_pil = sd_model.numpy_to_pil(image_numpy)
            image_pil[0].save(f"./results/sd/{prompt}/image.JPEG")

    elif sys.argv[1] == "SeedSelect":
        # run SeedSelect
        init_seed = torch.randn(shape, device=sd_model.device, dtype=clip_centroid.dtype).to(
                CUDA) * sd_model.scheduler.init_noise_sigma
        for i in range(2):
            ### hyper-parameters
            n_optimization_iters = 30
            lr = 0.1
            ###
            init_seed = torch.randn(shape, device=sd_model.device, dtype=clip_centroid.dtype).to(
                CUDA) * sd_model.scheduler.init_noise_sigma

            init_seed = optimize_seed(init_seed, prompt, sd_model, 
                        clip_model, clip_transform.transforms, 
                        clip_centroid, vae_centroid,
                        i,
                        n_iters=n_optimization_iters, guidance_scale=7.5, lr=lr)
        
    elif sys.argv[1] == "GCMI":
        batch = 2
        shape = (batch, sd_model.unet.in_channels, height // sd_model.vae_scale_factor, width // sd_model.vae_scale_factor)
        print(shape)
        init_seed = torch.randn(shape, device=sd_model.device, dtype=clip_centroid.dtype).to(
                CUDA) * sd_model.scheduler.init_noise_sigma
        ### hyper-parameters
        n_optimization_iters = 30
        lr = 0.1
        ###
        init_seed = torch.randn(shape, device=sd_model.device, dtype=clip_centroid.dtype).to(
            CUDA) * sd_model.scheduler.init_noise_sigma

        init_seed = submod_optimize_seed(batch, init_seed, prompt, sd_model, 
                    clip_model, clip_transform.transforms, 
                    clip_centroid, vae_centroid,
                    n_iters=n_optimization_iters, guidance_scale=7.5, lr=lr)
        
    # elif sys.argv[1] == "NAO_SeedSelect":
    #     # run SeedSelect initialized with NAO seeds
    #     all_real_x_t = image_folder_to_ddim(img_folder, prompt, sd_model)
    #     p1 = all_real_x_t[4].reshape((1, 4, 64, 64)).to("cuda")
    #     p2 = all_real_x_t[3].reshape((1, 4, 64, 64)).to("cuda")
    #     dim = 4 * 64 * 64

    #     ### hyper-parameters
    #     n_points = 50  # number of points between a centroid to an inversion
    #     n_optimization_iters = 5
    #     lr = 0.05
    #     p1 = p1.reshape(1, -1).T
    #     p2 = p2.reshape(1, -1).T
    #     eps = 2 * (torch.norm(p1 - p2) / n_points)
    #     ###

    #     log_chi_dist = lambda x: ((dim - 1) * torch.log(x) - 0.5 * torch.pow(x, 2)) - (
    #             (0.5 * dim - 1) * torch.log(torch.tensor(2.0)) + torch.lgamma(torch.tensor(dim / 2)))
    #     c, paths_ls = norm_aware_centroid_optimization(log_chi_dist, [p1, p2], n_points, eps, init_c=slerp(0.5, p1, p2))
    #     for path in paths_ls:
    #         for seed_i in path:
    #             optimize_seed(seed_i, prompt, sd_model, clip_model, clip_transform.transforms, clip_centroid,
    #                           vae_centroid, n_iters=n_optimization_iters, guidance_scale=1, lr=lr,
    #                           show_first_image=False)
