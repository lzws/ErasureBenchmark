import argparse
import os

import pandas as pd
import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from SalUn.eval_scripts.generate_images import SalUnGenerate
from AdvUnlearn.ATPipeline import generate_images as AdvGenerate
from SLD.SLDGenerate import generate_images as SLDGenerate
from SelfDiscover.generate.generate_images import generate_images as SelfDiscoverGenerate 
from UCE.generate_images import generate_images as UCEGenerate
from FMN.generate_images import generate_images as FMNGenerate



def SalUn_generate(erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = f'model/SalUn/nsfw/{erase_target}/nsfw-diffusers.pt'
    model_path = os.path.join(project_path, model_relative_path)
    if os.path.isfile(model_path):
        SalUnGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://github.com/OPTML-Group/Unlearn-Saliency.")


# AdvUnlearn
def AdvUnlearn_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = f'model/AdvUnlearn/nsfw/{erase_target}_unlearned'
    model_path = os.path.join(project_path, model_relative_path)
    # 这个是整个text encoder的文件夹
    if os.path.isfile(model_path)==False:
        AdvGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://drive.google.com/drive/folders/1Nf-EJ2W3CsZwpc5blZFi7tm7o1wEiTg4")

def SLD_generate(erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps):
    if erase_target == 'Nudity':
        pass
    if erase_target == 'Violence':
        erase_target = 'violent, blood, fighting'
    if erase_target == 'Disturrbing':
        erase_target = 'distorted faces, human flesh, broken bones'
    if erase_target == 'Hate':
        erase_target = 'hateful, defaming a race, holocaust scenes'
    SLDGenerate(erase_target,evaluation_task=evaluation_task, prompts_path=prompts_path, save_path=save_path, method=method, device=device,ddim_steps=steps, num_samples=1)
    
def SelfDiscover_generate(evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = "model/SelfDiscover/nsfw/nudity/checkpoints"
    model_path = os.path.join(project_path, model_relative_path)
    if os.path.isdir(model_path):
        SelfDiscoverGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://github.com/hangligit/InterpretDiffusion")

def UCE_generate(evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = "model/UCE/nsfw/nudity/erased-nudity-towards_uncond-preserve_false-sd_1_4-method_replace.pt"
    model_path = os.path.join(project_path, model_relative_path)

    if os.path.isfile(model_path):
        UCEGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://drive.google.com/drive/folders/1Nf-EJ2W3CsZwpc5blZFi7tm7o1wEiTg4")    

def FMN_generate(erase_target, evaluation_task,version_XL, project_path, prompts_path, save_path, device, steps):
    if version_XL:
        model_relative_path = f'model/FMN/{erase_target}_XL'
    else:
        model_relative_path = f'model/FMN/{erase_target}'
    model_path = os.path.join(project_path, model_relative_path)
    if not os.path.isdir(model_path):
        print('please train the model through EraseConceptBenchmark/method/train_sh/Nudity/FMN.sh.')
        print('All training related hyperparameters can be found here: EraseConceptBenchmark/model/FMN/configs/attn_nudity.yaml')
    else:
        FMNGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)

def NP_generate(evaluation_task, project_path, prompts_path, save_path, device, steps):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
    pipe.safety_checker = None
    negative_prompt = ["nudity"]
    df = pd.read_csv(prompts_path)
    start_index = 0
    for _,row in df.iterrows():
        if _ < start_index:
            continue
        prompt = [str(row.prompt)]
        seed = row.seed
        if evaluation_task == "i2p":
            case_number = row.case_number
        elif evaluation_task == "coco":
            case_number = row.image_id
        
        generator = torch.manual_seed(seed)
        image = pipe(
            prompt=prompt, generator = generator, guidance_scale=7.5,
            negative_prompt=negative_prompt,
            num_inference_steps=steps
        ).images[0]
        image.save(f"{save_path}/{case_number}.png")


if __name__ == "__main__":
    erase_concept = 'nudity'
    prompts_path = '/EraseConceptBenchmark/dataset/i2p_4703.csv'
    save_path = '/EraseConceptBenchmark/image_result/nsfw/i2p/SaiUn'

    SalUnGenerate(erase_concept=erase_concept, prompts_path=prompts_path, save_path=save_path)
    
