import os
import random
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import openai
import regex as re
import requests
import torch
from clip_retrieval.clip_client import ClipClient
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import DPMSolverMultistepScheduler

def retrieve(class_prompt, class_images_dir, num_class_images, save_images=False):
    factor = 1.5
    num_images = int(factor * num_class_images)
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion_400m",
        num_images=num_images,
        aesthetic_weight=0.1,
    )

    os.makedirs(f"{class_images_dir}/images", exist_ok=True)
    if len(list(Path(f"{class_images_dir}/images").iterdir())) >= num_class_images:
        return

    while True:
        class_images = client.query(text=class_prompt)
        if len(class_images) >= factor * num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(factor * num_images)
            client = ClipClient(
                url="https://knn.laion.ai/knn-service",
                indice_name="laion_400m",
                num_images=num_images,
                aesthetic_weight=0.1,
            )

    count = 0
    total = 0
    pbar = tqdm(desc="downloading real regularization images", total=num_class_images)

    if save_images:
        with open(f"{class_images_dir}/caption.txt", "w") as f1, open(
            f"{class_images_dir}/urls.txt", "w"
        ) as f2, open(f"{class_images_dir}/images.txt", "w") as f3:
            while total < num_class_images:
                images = class_images[count]
                count += 1
                try:
                    img = requests.get(images["url"])
                    if img.status_code == 200:
                        _ = Image.open(BytesIO(img.content))
                        with open(f"{class_images_dir}/images/{total}.jpg", "wb") as f:
                            f.write(img.content)
                        f1.write(images["caption"] + "\n")
                        f2.write(images["url"] + "\n")
                        f3.write(f"{class_images_dir}/images/{total}.jpg" + "\n")
                        total += 1
                        pbar.update(1)
                    else:
                        continue
                except:
                    continue
    else:
        with open(f"{class_images_dir}/caption.txt", "w") as f1:
            while count < num_class_images:
                images = class_images[count]
                count += 1
                f1.write(images["caption"] + "\n")
                pbar.update(1)
    return


retrieve(
            "people, body",
            "/home/users/diffusion/project/EraseConceptBenchmark/method/AC/diffusers/gcc_test",
            200,
            save_images=False,
        )