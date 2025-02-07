# nohup python /home/users/diffusion/project/EraseConceptBenchmark/method/generate.py > /home/users/diffusion/project/EraseConceptBenchmark/method/z_log/generate.log 2>&1 &

from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import csv
import os
from torch import Generator
from cd_def import *
import datetime
from cd_def import ESD_generate, SafeGen_generate, SPM_generate, MACE_generate, AC_generate, SD_generate


project_path = "/home/users/diffusion/project/EraseConceptBenchmark"
save_image_path = "/home/users/diffusion/project/EraseConceptBenchmark/image_result"
steps = 40
device = "cuda:1"
evaluation_task = "i2p" # select:["i2p","coco","4chan", "Lexica","Template"]
erase_target = "Violent" # select:["Nudity","Violent","Disturbing","Hate"]
method = "ESD-x"
# select:["SD1-4", "SD2-1", "SDXL", "SD3", "SD3-5"]
# select:["ESD-u", "ESD-x", "SPM", "AC", "SafeGen", "MACE"]
version_XL = False


if evaluation_task == "i2p":
    prompts_path = "/home/users/diffusion/project/EraseConceptBenchmark/dataset/i2p_4703.csv"
    if version_XL:
        save_path = os.path.join(save_image_path,erase_target,"nsfw/i2p_XL",method)
    else:
        save_path = os.path.join(save_image_path,erase_target,"nsfw/i2p",method)
    if not os.path.isdir(save_path):
	    os.makedirs(save_path, exist_ok=True)
elif evaluation_task=="coco":
    prompts_path = "/home/users/diffusion/project/EraseConceptBenchmark/dataset/coco_10k.csv"
    if version_XL:
        save_path = os.path.join(save_image_path,erase_target+"/coco_10k_XL",method)
    else:
        save_path = os.path.join(save_image_path,erase_target+"/coco_10k",method)
    if not os.path.isdir(save_path):
	    os.makedirs(save_path, exist_ok=True)
elif evaluation_task in ["4chan", "Lexica","Template"]:
    prompts_path = "/home/users/diffusion/project/EraseConceptBenchmark/dataset/"+evaluation_task+".csv"
    save_path = os.path.join(save_image_path,erase_target,"nsfw",evaluation_task,method)
    if not os.path.isdir(save_path):
	    os.makedirs(save_path, exist_ok=True)


# generate
if method in ["ESD-u", "ESD-x"]:
    ESD_generate(erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps)
elif method =="SafeGen":
    SafeGen_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method =="SPM":
    SPM_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method =="MACE":
    MACE_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method =="AC":
    AC_generate(erase_target, evaluation_task, version_XL, project_path, prompts_path, save_path, device, steps)
elif method in ["SD1-4", "SD2-1", "SDXL","SD3", "SD3-5"]:
    SD_generate(evaluation_task, method, project_path, prompts_path, save_path, device, steps)



