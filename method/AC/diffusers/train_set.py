from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from huggingface_hub import HfApi, create_repo
from model_pipeline import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionPipeline,
    set_use_memory_efficient_attention_xformers,
)
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from utils import (
    CustomDiffusionDataset,
    PromptDataset,
    collate_fn,
    filter,
    getanchorprompts,
    retrieve,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
# from diffusers.models.cross_attention import CrossAttention
from diffusers.models.attention import Attention as CrossAttention
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda:0")
class_images_dir = "/home/users/diffusion/project/EraseConceptBenchmark/method/AC/diffusers/data/nudity_XL"

num_new_images = 1000
with open(os.path.join(class_images_dir, "caption.txt")) as f:
    class_prompt_collection = [x.strip() for x in f.readlines()]

sample_dataset = PromptDataset(class_prompt_collection, num_new_images)
sample_dataloader = torch.utils.data.DataLoader(
    sample_dataset, batch_size=4
)

for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
            ):
    with open(f"{class_images_dir}/caption.txt", "a") as f1, open(f"{class_images_dir}/images.txt", "a") as f2:
        images = pipe(
            example["prompt"],
            num_inference_steps=40,
            # guidance_scale=6.0,
            # eta=1.0,
        ).images

        for i, image in enumerate(images):
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = (f"{class_images_dir}/images/{example['index'][i]}-{hash_image}.jpg")
            image.save(image_filename)
            f2.write(str(image_filename) + "\n")
        f1.write("\n".join(example["prompt"]) + "\n")