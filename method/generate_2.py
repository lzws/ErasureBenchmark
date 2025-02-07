# nohup python /home/users/diffusion/project/EraseConceptBenchmark/method/generate.py > /home/users/diffusion/project/EraseConceptBenchmark/method/generate.log 2>&1 &

from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import csv
import os
from torch import Generator

import datetime
# from cd_def import ESD_generate, SafeGen_generate, SPM_generate
from lzw_def import AdvUnlearn_generate, SalUn_generate, SLD_generate, SelfDiscover_generate, UCE_generate, FMN_generate, NP_generate

project_path = "/home/users/diffusion/project/EraseConceptBenchmark"
save_image_path = "/home/users/diffusion/project/EraseConceptBenchmark/image_result"

steps = 40
device = "cuda:0"
evaluation_task = "4chan" # select:["i2p","coco","4chan", "Lexica","Template"]
erase_target = "Nudity" # select:["Nudity","Violent","Disturbing","Hate"]
method = "FMN" # select:["AdvUnlearn","SalUn","SLD-Max","SLD-Strong","SLD-Medium","SelfDiscover","FMN","NP"]
version_XL = False
# select:["AdvUnlearn","SalUn","SLD-Max","SLD-Strong","SLD-Medium","SelfDiscover","FMN","NP"]


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
    ESD_generate(evaluation_task, method, project_path, prompts_path, save_path, device, steps)
elif method =="SafeGen":
    SafeGen_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method =="SPM":
    SPM_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method == "AdvUnlearn":
    AdvUnlearn_generate(erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method == "SalUn":
    SalUn_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method in ["SLD-Max","SLD-Strong","SLD-Medium"] :
    SLD_generate(erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps)
elif method == "SelfDiscover":
    SelfDiscover_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method == "UCE":
    UCE_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
elif method == "FMN":
    FMN_generate(erase_target, evaluation_task, version_XL, project_path, prompts_path, save_path, device, steps)
elif method == "NP":
    NP_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
else:
    raise ValueError()


# nohup python lzw_generate.py > z_log/generate_nudity_FMN_4chan.log 2>&1 &
# nohup python generate.py > z_log/generate_SD3-5_i2p.log 2>&1 &
# nohup python lzw_generate.py > FMN_versionXL_coco.log 2>&1 &