MODEL_NAME="CompVis/stable-diffusion-v1-4"
CONFIG_DIR="EraseConceptBenchmark/model/FMN/configs/attn_nudity.yaml"
OUTPUT_DIR="EraseConceptBenchmark/model/FMN/nsfw/nudity"

echo "FMNN requires the ten target images to forget target concept"
echo "All training related hyperparameters can be found here: EraseConceptBenchmark/model/FMN/configs/attn_nudity.yaml"

python EraseConceptBenchmark/method/FMN/run.py $CONFIG_DIR


echo "done! please check: $OUTPUT_DIR"
