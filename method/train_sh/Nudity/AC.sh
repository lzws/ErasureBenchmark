MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="/home/users/diffusion/project/EraseConceptBenchmark/model/AC/nsfw/nudity"
OUTPUT_DIR="/home/users/diffusion/project/EraseConceptBenchmark/method/AC/diffusers/log/nudity_XL"

echo "This method requires the tools to crawl prompt data in Laion-400m."
echo "We close it and make our own prompt dataset and put it in EraseConceptBenchmark/method/AC/diffusers/data/samples_nudity/caption.txt"

accelerate launch /home/users/diffusion/project/EraseConceptBenchmark/method/AC/diffusers/train.py \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --output_dir="$OUTPUT_DIR" \
    --class_data_dir=/home/users/diffusion/project/EraseConceptBenchmark/method/AC/diffusers/data/nudity_XL/ \
    --class_prompt="people, body" \
    --caption_target="nudity, nsfw" \
    --concept_type="nudity" \
    --resolution=512 \
    --train_batch_size=8 \
    --learning_rate=2e-7 \
    --max_train_steps=800 \
    --scale_lr \
    --hflip \
    --parameter_group="full-weight" \
    --enable_xformers_memory_efficient_attention

echo "done! please check: $OUTPUT_DIR"
