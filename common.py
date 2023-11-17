import cv2
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from utils.misc import load_config
from omegaconf import OmegaConf
import torch
from PIL import Image
import numpy as np
import os
from mvdiffusion.data.single_image_dataset import SingleImageDataset


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool


def get_config():
    # parse YAML config to OmegaConf
    cfg = load_config("./configs/mvdiffusion-joint-ortho-6views.yaml")
    # print(cfg)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)
    return cfg


def preprocess_data(single_image, crop_size):
    dataset = SingleImageDataset(
        root_dir="",
        num_views=6,
        img_wh=[256, 256],
        bg_color="white",
        crop_size=crop_size,
        single_image=single_image,
    )
    return dataset[0]


def tensor2pil(tensor):
    img = (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .squeeze(0)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    img = Image.fromarray(img)
    img = expand2square(img, (127, 127, 127, 0))
    return img


def save_image(tensor):
    ndarr = (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    return ndarr


def save_image_to_disk(tensor, fp):
    ndarr = (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
