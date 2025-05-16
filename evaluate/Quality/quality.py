import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd
import lpips
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import ImageReward as RM

def get_clip_score(image_path, text, model, processor, device):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
    return logits_per_image

def compute_clip_score(prompts_path, save_image_path, method, version_ec, erase_target, device, start_index=0):
    model_path = "clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    if method in ["SD1-4", "SD2-1", "SDXL","SD3", "SD3-5"]:
        image_path = os.path.join(save_image_path, method, "coco-10k")
    else:
        image_path = os.path.join(save_image_path, method, version_ec, erase_target, "coco-10k")

    image_type = 0
    if os.path.isfile(os.path.join(image_path, "73.png")):
        image_type = 1
    elif os.path.isfile(os.path.join(image_path, "73_0.png")):
        image_type = 2
    else:
        print("The name of the image does not comply with the regulations!")
    print(f"image_type:{image_type} (type-1:73.png; type-2:73_0.png).")

    clip_score = 0
    df = pd.read_csv(prompts_path)
    
    for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df)-start_index, desc="Processing images"):
        number = str(row.image_id)
        prompt = str(row.prompt)
        
        # Construct image path based on passed arguments
        if image_type == 1:
            img_path = os.path.join(image_path, f"{number}.png")
        elif image_type == 2:
            img_path = os.path.join(image_path, f"{number}_0.png")

        result = 0
        if os.path.isfile(img_path):
            result = get_clip_score(img_path, prompt, model, processor, device)
            clip_score += result.item()
        
        if index == 0:
            print(f"index is {index}, result is {result}")

    item_count = len(os.listdir(image_path))
    print(f"CLIP_Score: sum: {clip_score}, number: {item_count}, avg(CLIP_Score): {clip_score / item_count}")

def trans(img_path, compare_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert('RGB')
    compare_img = Image.open(compare_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    compare_tensor = transform(compare_img).unsqueeze(0)
    return img_tensor, compare_tensor

def compute_LPIPS(prompts_path, compare_img_path, save_image_path, method, version_ec, erase_target, device, start_index=0):
    if method in ["SD1-4", "SD2-1", "SDXL","SD3", "SD3-5"]:
        image_path = os.path.join(save_image_path, method, "coco-10k")
    else:
        image_path = os.path.join(save_image_path, method, version_ec, erase_target, "coco-10k")
    loss_fn = lpips.LPIPS(net='alex', version=0.1).to(device)
    
    image_type = 0
    if os.path.isfile(os.path.join(image_path, "73.png")):
        image_type = 1
    elif os.path.isfile(os.path.join(image_path, "73_0.png")):
        image_type = 2
    else:
        print("The name of the image does not comply with the regulations!")
    print(f"image_type:{image_type} (type-1:73.png; type-2:73_0.png).")

    df = pd.read_csv(prompts_path)
    total_lpips_distance = 0
    average_lpips_distance = 0
    
    for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df)-start_index, desc="Processing images"):
        number = str(row.image_id)
        
        # Construct image path based on passed arguments
        if image_type == 1:
            img_path = os.path.join(image_path, f"{number}.png")
        elif image_type == 2:
            img_path = os.path.join(image_path, f"{number}_0.png")
        compare_path = os.path.join(compare_img_path,f"{number}.jpg")

        img_tensor, compare_tensor = trans(img_path, compare_path)
        img_tensor = img_tensor.to(device)
        compare_tensor = compare_tensor.to(device)
        current_lpips_distance = loss_fn.forward(img_tensor, compare_tensor)
        total_lpips_distance = total_lpips_distance + current_lpips_distance
 
        if index == 0:
            print(f"index is {index}, result is {current_lpips_distance}")

    item_count = len(os.listdir(image_path))
    average_lpips_distance = float(total_lpips_distance) / item_count
    print(f"LPIPS: sum: {total_lpips_distance}, number: {item_count}, avg(CLIP_Score): {average_lpips_distance}")

def compute_reward(prompts_path, save_image_path, method, version_ec, erase_target, device, start_index=0):
    model = RM.load("ImageReward-v1.0").to(device)
    if method in ["SD1-4", "SD2-1", "SDXL","SD3", "SD3-5"]:
        image_path = os.path.join(save_image_path, method, "coco-10k")
    else:
        image_path = os.path.join(save_image_path, method, version_ec, erase_target, "coco-10k")

    image_type = 0
    if os.path.isfile(os.path.join(image_path, "73.png")):
        image_type = 1
    elif os.path.isfile(os.path.join(image_path, "73_0.png")):
        image_type = 2
    else:
        print("The name of the image does not comply with the regulations!")
    print(f"image_type:{image_type} (type-1:73.png; type-2:73_0.png).")

    score = 0
    df = pd.read_csv(prompts_path)
    with torch.no_grad():
        for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df)-start_index, desc="Processing images"):
            number = str(row.image_id)
            prompt = str(row.prompt)
            
            # Construct image path based on passed arguments
            if image_type == 1:
                img_path = os.path.join(image_path, f"{number}.png")
            elif image_type == 2:
                img_path = os.path.join(image_path, f"{number}_0.png")

            result = 0
            if os.path.isfile(img_path):
                result = model.score(prompt, img_path)
                score += result
            
            if index == 0:
                print(f"index is {index}, result is {result}")

    item_count = len(os.listdir(image_path))
    print(f"Reward_Score: sum: {score}, number: {item_count}, avg(CLIP_Score): {score / item_count}")


def main():
    parser = argparse.ArgumentParser(description="CLIP score computation")
    parser.add_argument('--compute', type=str, choices=["CLIP-score", "LPIPS", "Reward"], required=True, help="Path to the pre-trained CLIP model")
    parser.add_argument('--prompts_path', type=str, required=True, help="Path to the CSV file containing prompts and image IDs")
    parser.add_argument('--compare_img_path', type=str, default="/home/users/diffusion/project/EraseConceptBenchmark/evaluate/coco_10k_real", help="Path to save the compared images")
    parser.add_argument('--save_image_path', type=str, default="/home/users/diffusion/project/EraseConceptBenchmark/image_result", help="Path to save the generated images")
    parser.add_argument('--method', type=str, required=True, help="Method for concept erasure")
    parser.add_argument('--version_ec', type=str, choices=["keywords-less", "keywords-more", "all-nsfw"], required=True, help="Version of the concept erasure method")
    parser.add_argument('--erase_target', type=str, choices=["Nudity", "Violent", "Disturbing", "Hate"], required=True, help="Concept to erase")
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cuda:0', 'cuda:1', 'cpu'], help="Device to run the model on (default: cuda:1)")
    parser.add_argument('--start_index', type=int, default=0, help="Index to start processing from in the CSV file (default: 0)")
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.compute == "CLIP-score":
        compute_clip_score(
            args.prompts_path, 
            args.save_image_path, 
            args.method, 
            args.version_ec, 
            args.erase_target, 
            device, 
            start_index=args.start_index
        )
    elif args.compute == "LPIPS":
        compute_LPIPS(
            args.prompts_path, 
            args.compare_img_path,
            args.save_image_path, 
            args.method, 
            args.version_ec, 
            args.erase_target, 
            device, 
            start_index=args.start_index
        )
    elif args.compute == "Reward":
        compute_reward(
            args.prompts_path, 
            args.save_image_path, 
            args.method, 
            args.version_ec, 
            args.erase_target, 
            device, 
            start_index=args.start_index
        )

if __name__ == "__main__":
    main()
