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


def SalUn_generate(version_ec, erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = f'model/SalUn/{erase_target}/{erase_target}-{version_ec}-diffusers.pt'
    if erase_target == 'all-nsfw':
        model_relative_path = f'model/SalUn/{erase_target}/{erase_target}-diffusers.pt'
    model_path = os.path.join(project_path, model_relative_path)
    if evaluation_task == 'generalization':
        model_path = f'/shark/zhiwen/benchmark/EraseBenchmark/method/SalUn/erase-models/{erase_target}/{erase_target}-200-diffusers.pt'
    if os.path.isfile(model_path):
        SalUnGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://github.com/OPTML-Group/Unlearn-Saliency.")


# AdvUnlearn
def AdvUnlearn_generate(version_ec,  erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):
    model_relative_path = f'model/AdvUnlearn/{erase_target}/{version_ec}'
    model_path = os.path.join(project_path, model_relative_path)
    if evaluation_task == 'generalization':
        model_path = f'/shark/zhiwen/benchmark/EraseBenchmark/method/AdvUnlearn/results/models/{erase_target}'
            # 这个是整个text encoder的文件夹
    if os.path.isfile(model_path)==False:
        AdvGenerate(version_ec=version_ec, erase_target=erase_target, evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://drive.google.com/drive/folders/1Nf-EJ2W3CsZwpc5blZFi7tm7o1wEiTg4")

def SLD_generate(version_ec,erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps):

    if evaluation_task == 'generalization':
        pass
    else:
        erase_target = unsafe_concepts[erase_target][version_ec]
    SLDGenerate(erase_target,evaluation_task=evaluation_task, prompts_path=prompts_path, save_path=save_path, method=method, device=device,ddim_steps=steps, num_samples=1)
    
def SelfDiscover_generate(version_ec, erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps):
    # model_relative_path = f"model/SelfDiscover/nsfw/{erase_target}/checkpoints"
    model_relative_path = f'model/SelfDiscover/{erase_target}/{version_ec}'
    if erase_target == 'all-nsfw':
        model_relative_path = f'model/SelfDiscover/{erase_target}'
    model_path = os.path.join(project_path, model_relative_path)
    if evaluation_task == 'generalization':
        model_path = f'/shark/zhiwen/benchmark/EraseBenchmark/method/SelfDiscover/exps/{erase_target}'
    if os.path.isdir(model_path):
        SelfDiscoverGenerate(erase_target=erase_target,evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://github.com/hangligit/InterpretDiffusion")

def UCE_generate(version_ec, erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps):

    model_relative_path = f'model/UCE/{erase_target}/{version_ec}/{erase_target}-{version_ec}.pt'
    if erase_target == 'all-nsfw':
        model_relative_path = f'model/UCE/{erase_target}/{version_ec}/{erase_target}-{version_ec}.pt'
    model_path = os.path.join(project_path, model_relative_path)
    if evaluation_task == 'generalization':
        model_path = f'/shark/zhiwen/benchmark/EraseBenchmark/method/UCE/models/{erase_target}/erased-{erase_target}.pt'
    print('model_path: ',model_path)
    # 这个是整个text encoder的文件夹
    if os.path.isfile(model_path):
        UCEGenerate(erase_target=erase_target, evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)
    else:
        print("please download the model from https://drive.google.com/drive/folders/1Nf-EJ2W3CsZwpc5blZFi7tm7o1wEiTg4")    

def FMN_generate(version_ec, erase_target, evaluation_task,version_XL, project_path, prompts_path, save_path, device, steps):
    if erase_target == 'all-nsfw':
        model_relative_path = f'model/FMN/{erase_target}'
    else:
        model_relative_path = f'model/FMN/{erase_target}/{version_ec}'
    model_path = os.path.join(project_path, model_relative_path)
    if evaluation_task == 'generalization':
        model_path = f'/shark/zhiwen/benchmark/EraseBenchmark/method/FMN/exps_attn/{erase_target}200'
    print('model_path: ',model_path)
    if not os.path.isdir(model_path):
        print('please train the model through EraseConceptBenchmark/method/train_sh/Nudity/FMN.sh.')
        print('All training related hyperparameters can be found here: EraseConceptBenchmark/model/FMN/configs/attn_nudity.yaml')
    else:
        FMNGenerate(evaluation_task = evaluation_task, model_name = model_path, prompts_path = prompts_path, save_path = save_path, device = device, ddim_steps = steps, num_samples = 1)

def NP_generate(version_ec,erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps):

    # erase_target = unsafe_concepts[erase_target][version_ec]
    if evaluation_task != 'generalization':
        erase_target = unsafe_concepts[erase_target][version_ec]
    print(f'NP erase target:{erase_target}, save path: {save_path}')
    pipe = StableDiffusionPipeline.from_pretrained('/shark/zhiwen/benchmark/models/stable-diffusion-v1-4').to(device)
    pipe.safety_checker = None
    negative_prompt = [erase_target]
    df = pd.read_csv(prompts_path)
    start_index = 0
    for _,row in df.iterrows():
        if _ < start_index:
            continue
        prompt = [str(row.prompt)]
        seed = row.seed

        if evaluation_task == "coco":
            case_number = row.image_id
        else:
            case_number = row.case_number
        
        if evaluation_task == 'generalization':
            keyword = row.keyword
            os.makedirs(save_path+'/'+keyword, exist_ok=True)
        else:
            keyword = ""

        
        
        if os.path.isfile(save_path+'/'+str(case_number)+".png"):
            print(f"{case_number}.png exist!")
            continue
        
        generator = torch.manual_seed(seed)
        image = pipe(
            prompt=prompt, generator = generator, guidance_scale=7.5,
            negative_prompt=negative_prompt,
            num_inference_steps=steps
        ).images[0]
        if evaluation_task == 'generalization':
            image.save(f"{save_path}/{keyword}/{case_number}.png")
        else:
            image.save(f"{save_path}/{case_number}.png")
