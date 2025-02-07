MODEL_NAME="CompVis/stable-diffusion-v1-4"

conda env create -f environment.yaml
conda activate sa-sd

nohup python main_forget.py -t --base configs/stable-diffusion/forget_nudity.yaml --gpus "0,1,2,3" --num_nodes 1 --finetune_from models/ldm/sd-v1-4-full-ema.ckpt > SA_training.log 2>&1 &


