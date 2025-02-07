import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def generate_images(evaluation_task, model_name, prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    pipe.safety_checker = None

    print(f'FMN task：{evaluation_task} model name: {model_name} save path: {save_path}')

    df = pd.read_csv(prompts_path)
    for _,row in df.iterrows():

        prompt = [str(row.prompt)]
        seed = row.seed

        if evaluation_task == "i2p":
            case_number = row.case_number
        elif evaluation_task == "coco":
            case_number = row.image_id

        if _ < from_case:
            continue
        generator = torch.manual_seed(seed) 

        image = pipe(
            prompt=prompt, generator = generator, guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            height=image_size,
            width=image_size,
            num_inference_steps=ddim_steps
        ).images[0]

        image.save(f"{save_path}/{case_number}_{0}.png")
