TRAIN_METHOD="noxattn" # pick from following 'xattn','noxattn', 'selfattn', 'full'
MODEL_SAVE_PATH="/home/users/diffusion/project/EraseConceptBenchmark/model/ESD/nsfw/Hate"
DEVICE="cuda:0"

python /home/users/diffusion/project/EraseConceptBenchmark/method/ESD/esd_diffusers.py --erase_concept 'hateful, defaming a race, holocaust scenes' --train_method "$TRAIN_METHOD" --save_path "$MODEL_SAVE_PATH" --device "$DEVICE"