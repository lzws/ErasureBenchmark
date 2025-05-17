# nohup python /home/users/diffusion/project/EraseConceptBenchmark/method/generate.py > /home/users/diffusion/project/EraseConceptBenchmark/method/generate.log 2>&1 &

from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import csv
import os
from torch import Generator
import argparse
import datetime
# from cd_def import ESD_generate, SafeGen_generate, SPM_generate
from method_def import *




# select:["AdvUnlearn","SalUn","SLD-Max","SLD-Strong","SLD-Medium","SelfDiscover","FMN","NP"]



def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate images with content erasure methods.")
    parser.add_argument('--steps', type=int, default=40, help='Number of steps for the diffusion process')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device for computation, e.g., "cuda:0" or "cpu"')
    parser.add_argument('--evaluation_task', type=str, choices=["i2p", "coco", "4chan", "Lexica", "Template"], default="i2p", help='Evaluation task')
    parser.add_argument('--erase_target', type=str, choices=["Nudity", "Violent", "Disturbing", "Hate"], default="Nudity", help='Target to erase from the generated image')
    parser.add_argument('--method', type=str, choices=["ESD-u", "ESD-x", "SPM", "SafeGen", "MACE", "AC", "SD1-4", "SD2-1", "SDXL", "SD3", "SD3-5","AdvUnlearn","SalUn","SLD-Max","SLD-Strong","SLD-Medium","SelfDiscover","FMN","NP","UCE"], default="AC", help='Erasure method')
    parser.add_argument('--version_XL', action='store_true', help='Flag to use the XL version of the model (if applicable)')
    return parser.parse_args()

def main(args):
    project_path = args.project_path
    save_image_path = args.save_image_path
    steps = args.steps
    device = args.device
    evaluation_task = args.evaluation_task
    erase_target = args.erase_target
    method = args.method
    version_ec = args.version_ec
    version_XL = args.version_XL
    version_img = args.version_img

    # Determine the prompts path
    if evaluation_task == "i2p":
        prompts_path = os.path.join(project_path, "dataset", "i2p_4703.csv")
    elif evaluation_task in ["4chan", "Lexica", "Template"]:
        prompts_path = os.path.join(project_path, "dataset", f"{evaluation_task}.csv")
    elif evaluation_task == "coco":
        prompts_path = os.path.join(project_path, "dataset", "coco_10k.csv")
    elif evaluation_task == "RAB2":
        prompts_path = os.path.join(project_path, "dataset", "adv_prompts", f"{erase_target}_adv_prompts.csv")
    elif evaluation_task == "generalization":
        prompts_path = os.path.join(project_path, "dataset", "dataset", "generalization_prompts", f"{erase_target}.csv")

    # Determine the save path
    if method in ["SD1-4", "SD2-1", "SDXL", "SD3", "SD3-5"]:
        save_path = os.path.join(save_image_path, method, evaluation_task)
        if evaluation_task == 'RAB2':
            save_path = os.path.join(save_image_path, method, evaluation_task, erase_target)
        if evaluation_task == 'generalization':
            save_path = os.path.join(save_image_path, method, evaluation_task, erase_target)
    else:
        if version_XL:
            save_path = os.path.join(save_image_path, method, version_ec, erase_target, f"{evaluation_task}XL")
        else:
            
            save_path = os.path.join(save_image_path, method, version_ec, erase_target, evaluation_task)
            if erase_target == 'all-nsfw':
                if version_ec == 'keywords-more':
                    save_path = os.path.join(save_image_path, method, version_ec, erase_target, evaluation_task)
                else:
                    save_path = os.path.join(save_image_path, method, erase_target, evaluation_task)

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)


    # generate
    if method in ["ESD-u", "ESD-x"]:
        ESD_generate(version_ec, erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps)
    elif method =="SafeGen":
        SafeGen_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method =="SPM":
        SPM_generate(evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method == "AdvUnlearn":
        AdvUnlearn_generate(version_ec,erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method == "SalUn":
        SalUn_generate(version_ec, erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method in ["SLD-Max","SLD-Strong","SLD-Medium"] :
        SLD_generate(version_ec, erase_target, evaluation_task, method, project_path, prompts_path, save_path, device, steps)
        print('generate done !!!!!!!!!!')
    elif method == "SelfDiscover":
        SelfDiscover_generate(version_ec, erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method == "UCE":
        UCE_generate(version_ec, erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
        print('generate done !!!!!!!!!!')
    elif method == "FMN":
        FMN_generate(version_ec, erase_target, evaluation_task, version_XL, project_path, prompts_path, save_path, device, steps)
    elif method =="MACE":
        MACE_generate(erase_target, evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method =="AC":
        AC_generate(erase_target, evaluation_task, version_XL, project_path, prompts_path, save_path, device, steps)
    elif method == "NP":
        NP_generate(version_ec, erase_target,evaluation_task, project_path, prompts_path, save_path, device, steps)
    elif method in ["SD1-4", "SD2-1", "SDXL","SD3", "SD3-5"]:
        SD_generate(evaluation_task, method, project_path, prompts_path, save_path, device, steps)
    else:
        raise ValueError()
    print('generate done !!!!!!!!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation with concept erasure methods")

    # Arguments
    parser.add_argument("--project_path", type=str, default="/shark/zhiwen/benchmark/EraseBenchmark")
    parser.add_argument("--save_image_path", type=str, default="/shark/zhiwen/benchmark/EraseBenchmark/image_result", help="Path to save the generated images")
    parser.add_argument("--steps", type=int, default=40, help="Number of diffusion steps")
    parser.add_argument("--device", type=str, default="cuda:0", required=True, help="Device for computation (e.g., cuda:0, cpu)")
    parser.add_argument("--evaluation_task", type=str, choices=["i2p", "4chan", "Lexica", "Template", "coco", "RAB2","generalization"], required=True, help="Evaluation task")
    parser.add_argument("--erase_target", type=str, choices=["nudity","violence","horror","Nudity", "Violent", "Disturbing", "Hate", "Political", "all-nsfw"], required=True, help="Concept to erase")
    parser.add_argument("--method", type=str, choices=["ESD-u", "ESD-x", "SPM", "SafeGen", "MACE", "AC", "SD1-4", "SD2-1", "SDXL", "SD3", "SD3-5","AdvUnlearn","SalUn","SLD-Max","SLD-Strong","SLD-Medium","SelfDiscover","FMN","NP","UCE"], required=True, help="Method for concept erasure")
    parser.add_argument("--version_ec", required=True, help="Version of the concept erasure method")
    parser.add_argument("--version_XL", type=bool, default=False, help="Use XL version of the method (True/False)")
    parser.add_argument("--version_img", type=int, choices=[20, 200, 1000], default=20, help="Use XL version of the method (True/False)")

    args = parser.parse_args()

    main(args)

