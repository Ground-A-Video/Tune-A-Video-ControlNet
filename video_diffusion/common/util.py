import os
import sys
import copy
import inspect
import datetime
from typing import List, Tuple, Optional, Dict, Union
import torch
from tqdm import tqdm
import numpy as np

from video_diffusion.models.controlnet import ControlNetModel
from video_diffusion.models.controlnet_3d import ControlNetModel3D
from video_diffusion.pipelines.pipeline_tuneavideo_controlnet import MultiControlNetModel
from einops import rearrange

def glob_files(
    root_path: str,
    extensions: Tuple[str],
    recursive: bool = True,
    skip_hidden_directories: bool = True,
    max_directories: Optional[int] = None,
    max_files: Optional[int] = None,
    relative_path: bool = False,
) -> Tuple[List[str], bool, bool]:
    """glob files with specified extensions

    Args:
        root_path (str): _description_
        extensions (Tuple[str]): _description_
        recursive (bool, optional): _description_. Defaults to True.
        skip_hidden_directories (bool, optional): _description_. Defaults to True.
        max_directories (Optional[int], optional): max number of directories to search. Defaults to None.
        max_files (Optional[int], optional): max file number limit. Defaults to None.
        relative_path (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[List[str], bool, bool]: _description_
    """
    paths = []
    hit_max_directories = False
    hit_max_files = False
    for directory_idx, (directory, _, fnames) in enumerate(os.walk(root_path, followlinks=True)):
        if skip_hidden_directories and os.path.basename(directory).startswith("."):
            continue

        if max_directories is not None and directory_idx >= max_directories:
            hit_max_directories = True
            break

        paths += [
            os.path.join(directory, fname)
            for fname in sorted(fnames)
            if fname.lower().endswith(extensions)
        ]

        if not recursive:
            break

        if max_files is not None and len(paths) > max_files:
            hit_max_files = True
            paths = paths[:max_files]
            break

    if relative_path:
        paths = [os.path.relpath(p, root_path) for p in paths]

    return paths, hit_max_directories, hit_max_files


def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"


def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict

#### DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet, controlnet=None, c_image=None, controlnet_conditioning_scale=None):
    if isinstance(controlnet, ControlNetModel) or (isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet.nets[0], ControlNetModel)):
        ## 2D ControlNet model
        f = latents.shape[2]
        context_controlnet = torch.stack([context.clone() for _ in range(f)], dim=0)
        context_controlnet = rearrange(context_controlnet, 'a b c d -> (a b) c d')
        latents_controlnet = rearrange(latents.clone(), "b c f h w -> (b f) c h w")
        down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=latents_controlnet,
                    timestep=t,
                    encoder_hidden_states=context_controlnet,
                    controlnet_cond=c_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
        mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', f=f)
        down_block_res_samples = [rearrange(down_block_res_sample,'(b f) c h w -> b c f h w', f=f) for down_block_res_sample in down_block_res_samples]
        down_block_res_samples = tuple(down_block_res_samples)
    elif isinstance(controlnet, ControlNetModel3D) or (isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet.nets[0], ControlNetModel3D)):
        ## 3D ControlNet model
        down_block_res_samples, mid_block_res_sample = controlnet(
                        sample=latents,
                        timestep=t,
                        encoder_hidden_states=context,
                        controlnet_cond=c_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        return_dict=False,
                    )
        down_block_res_samples = tuple(down_block_res_samples)
    else:
        ## No ControlNet model
        mid_block_res_sample = None
        down_block_res_samples = None

    noise_pred = unet(latents, t, encoder_hidden_states=context,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, c_image, controlnet_conditioning_scale):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        if c_image is not None:
            noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, pipeline.controlnet, c_image, controlnet_conditioning_scale)
        else:
            noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent

#### Fix code to consider controlnet: "c_image", "pipeline.controlnet"
@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", c_image=None, controlnet_conditioning_scale=None):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, c_image, controlnet_conditioning_scale)
    return ddim_latents