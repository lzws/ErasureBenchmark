IMG_DIR=EraseConceptBenchmark/image_result/coco_10k_XL/FMN
DEVICE=cuda:0

echo $IMG_DIR
python -m pytorch_fid EraseConceptBenchmark/evaluate/coco_10k_real $IMG_DIR --device $DEVICE
