import os
import torch
from pathlib import Path
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusion3Pipeline
# from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from accelerate import PartialState, Accelerator
# from AC.diffusers.model_pipeline import CustomDiffusionPipeline
from ESD.utils.utils import *
# from SPM.generate_images import get_dataloader, infer_with_spm
from MACE.generate import MACE_generate_images
import torch

# from SPM.evaluate_task import main

def SD_generate(evaluation_task, method, project_path, prompts_path, save_path, device, steps):
    if method == "SD1-4":
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    elif method == "SD2-1":
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(device)
    elif method == "SDXL":
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)
    elif method == "SD3":
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",torch_dtype=torch.float16).to(device)
    elif method == "SD3-5":
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",torch_dtype=torch.bfloat16).to(device)

    pipe.safety_checker = None
    df = pd.read_csv(prompts_path)
    start_index = 3450
    for index, row in df.iloc[start_index:].iterrows():
        if evaluation_task == "i2p":
            number = row.case_number
        elif evaluation_task=="coco":
            number = row.image_id
        seed=row.seed
        generator = torch.manual_seed(seed)
        prompt = str(row.prompt)
        safe_image = pipe(prompt=prompt, generator=generator,num_inference_steps = steps).images[0]
        safe_image.save(save_path+'/'+str(number)+".png")


def ESD_generate(erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps):
    if method == "ESD-u":
        # model_relative_path = "diffusers-nudity-ESDu1-UNET.pt"
        model_relative_path = "esd-noxattn_1-epochs_200.pt"
    elif method == "ESD-x":
        model_relative_path = "esd-xattn_1-epochs_200.pt"
    model_path = os.path.join(project_path, "model/ESD/nsfw", erase_target, model_relative_path)
    if os.path.isfile(model_path):
        if method == "ESD-u":
            train_method = 'noxattn'
            diffuser = StableDiffuser(scheduler='DDIM').to(device)
            finetuner = FineTunedModel(diffuser, train_method=train_method)
            finetuner.load_state_dict(torch.load(model_path))
        elif method == "ESD-x":
            train_method = 'xattn'
            diffuser = StableDiffuser(scheduler='DDIM').to(device)
            finetuner = FineTunedModel(diffuser, train_method=train_method)
            finetuner.load_state_dict(torch.load(model_path))
        df = pd.read_csv(prompts_path)
        start_index = 0
        for index, row in df.iloc[start_index:].iterrows():
            if evaluation_task == "coco":
                number = row.image_id
            else:
                number = row.case_number
            seed=row.seed
            prompt = str(row.prompt)
            with finetuner:
                safe_image = diffuser(prompt,
                        img_size=512,n_steps=steps,n_imgs=1,
                        generator=torch.Generator().manual_seed(seed),
                        guidance_scale=7.5)[0][0]
            print(save_path+'/'+str(number)+".png")
            safe_image.save(save_path+'/'+str(number)+".png")
    else:
        print("please download the model from https://erasing.baulab.info/weights/esd_models.")

def SafeGen_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = "model/SafeGen/nsfw/nudity/SafeGen-Pretrained-Weights"
    model_path = os.path.join(project_path, model_relative_path)
    if not os.path.isdir(model_path):
        print("please download the model from https://huggingface.co/LetterJohn/SafeGen-Pretrained-Weights.")
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipeline.to(device)
        df = pd.read_csv(prompts_path)
        start_index = 0
        for index, row in df.iloc[start_index:].iterrows():
            if evaluation_task == "i2p":
                number = row.case_number
            elif evaluation_task=="coco":
                number = row.image_id
            seed=row.seed
            prompt = str(row.prompt)
            generator = torch.manual_seed(seed)
            safe_image = pipeline(prompt=prompt, generator=generator,num_inference_steps = steps).images[0]
            safe_image.save(save_path+'/'+str(number)+".png")

def SPM_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = "model/SPM/nsfw/nudity/nudity.safetensors"
    model_path = os.path.join(project_path, model_relative_path)
    if not os.path.isfile(model_path):
        print("please download the model from https://github.com/Con6924/SPM.")
    else:
        spm_paths = [Path(model_path)]
        print(spm_paths)
        dataloader = get_dataloader(task = evaluation_task, task_args = None, prompts_path = prompts_path, img_save_path = save_path, generation_cfg = None, num_processes=1)
        infer_with_spm(
            dataloader,
            spm_paths = spm_paths,
            matching_metric = 'clipcos_tokenuni',
            num_inference_steps = 40,
        )

def MACE_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = "model/MACE/nsfw/nudity/erase_explicit_content"
    model_path = os.path.join(project_path, model_relative_path)
    if not os.path.isdir(model_path):
        print("please download the model from https://github.com/Shilin-LU/MACE.")
    else:
        MACE_generate_images(evaluation_task, model_name=model_path, prompts_path=prompts_path, save_path=save_path, step=1, device=device, ddim_steps=40)

def AC_generate(erase_target, evaluation_task, version_XL, project_path, prompts_path, save_path, device, steps):
    if version_XL:
        model_relative_path = "model/AC/nsfw/nudity_XL/delta.bin"
    else:
        model_relative_path = "model/AC/nsfw/nudity/delta.bin"
    model_path = os.path.join(project_path, model_relative_path)
    if not os.path.isfile(model_path):
        print("please train the model through EraseConceptBenchmark/method/train_sh/Nudity/ac.sh.")
    else:
        print("please use diffusers==0.14.0")
        pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
        pipe.load_model(model_path)
        pipe.safety_checker = None
        df = pd.read_csv(prompts_path)
        start_index = 0
        for index, row in df.iloc[start_index:].iterrows():
            if evaluation_task == "i2p":
                number = row.case_number
            elif evaluation_task=="coco":
                number = row.image_id
            seed=row.seed
            prompt = str(row.prompt)
            generator = torch.manual_seed(seed)
            safe_image = pipe(prompt=prompt, generator=generator,num_inference_steps = steps, guidance_scale=6., eta=1.).images[0]
            safe_image.save(save_path+'/'+str(number)+".png")
    
def SafeClip_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = "model/SafeGen/nsfw/nudity/SafeGen-Pretrained-Weights"
    