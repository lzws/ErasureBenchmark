from .sld_pipeline import SLDPipeline

import os
import pandas as pd
import torch

def generate_images(erase_target, evaluation_task, prompts_path, save_path, method='SLD-Medium', device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0):
    
    sldpipe = SLDPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',safety_concept=erase_target).to(device)
    sldpipe.safety_checker = None

    print(f'{method} task: {evaluation_task} save path: {save_path}')

    df = pd.read_csv(prompts_path)
    for _,row in df.iterrows():

        prompt = [str(row.prompt)]
        seed = row.seed

        if evaluation_task == "coco":
            case_number = row.image_id
        else:
            case_number = row.case_number

        if _ < from_case:
            continue
        generator = torch.manual_seed(seed) 

        if method == 'SLD-Medium':
            sld_warmup_steps=10
            sld_guidance_scale=1000
            sld_threshold=0.01
            sld_momentum_scale=0.3
            sld_mom_beta=0.4
        elif method == "SLD-Strong":
            sld_warmup_steps=7
            sld_guidance_scale=2000
            sld_threshold=0.025
            sld_momentum_scale=0.5
            sld_mom_beta=0.7
        elif method == "SLD-Max":
            sld_warmup_steps=0
            sld_guidance_scale=5000
            sld_threshold=1.0
            sld_momentum_scale=0.5
            sld_mom_beta=0.7

        image = sldpipe(
            prompt=prompt, generator = generator, guidance_scale=guidance_scale,
            sld_warmup_steps=sld_warmup_steps,
            sld_guidance_scale=sld_guidance_scale,
            sld_threshold=sld_threshold,
            sld_momentum_scale=sld_momentum_scale,
            sld_mom_beta=sld_mom_beta,
            num_images_per_prompt=num_samples,
            height=image_size,
            width=image_size,
            num_inference_steps=ddim_steps
        ).images[0]

        image.save(f"{save_path}/{case_number}.png")

