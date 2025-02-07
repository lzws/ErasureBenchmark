# from .SelfDiscoverPipeline import SelfDiscoverPipeline
from .unet2dcondition import SelfUNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import os
import torch
import pandas as pd
import json
from .model import MLP
from .utils_model import load_weights
from PIL import Image

# device='cuda:1'
# basepath = '/home/users/zhiwen/project/'
# # model_version=basepath+"huggingfacemodels/stable-diffusion-v1-4/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
# model_version = '/home/users/diffusion/project/DiffusionModel/huggingfacemodels/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
# pipe = SelfDiscoverPipeline.from_pretrained(
#     model_version,
# ).to(device)
# pipe.safety_checker=None

# unet = SelfUNet2DConditionModel.from_pretrained(model_version, subfolder="unet").to(device)
# pipe.unet = unet
# mlp=MLP(input_dim=8, resolution=512//64).to(device)
# pipe.unet.set_controlnet(mlp)

# # mlp_path = 'checkpoints/unet.pth'
# mlp_path = '/home/users/diffusion/project/EraseConceptBenchmark/method/SelfDiscover/checkpoints/unet.pth'
# pipe.unet=load_weights(pipe.unet, mlp_path)

# concept_path = "/home/users/diffusion/project/EraseConceptBenchmark/method/SelfDiscover/checkpoints/concept_dict.json"
# # concept_path = "checkpoints/concept_dict.json"
# concept_dict = json.load(open(concept_path,"r"))

# condition=torch.zeros(1,8, device=device)
# condition[:,concept_dict['antisexual']]=1
# print(condition)

# prompts = ["a beautiful girl"]

# generator = torch.manual_seed(54689456)    # Seed generator to create the inital latent noise
# out = pipe(prompt=prompts, generator=generator, guidance_scale=7.5, controlnet_cond=condition).images[0]
# out.save('test.png')


def generate_images(evaluation_task, model_name, prompts_path, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0):
    
    tokenizer = CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="text_encoder")

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae")

    unet = SelfUNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet")

    # 输入维度要看config文件
    mlp_path = os.path.join(model_name, 'unet.pth')
    concept_path = os.path.join(model_name, 'concept_dict.json')
    concept_dict = json.load(open(concept_path,"r"))
    concept_nums = len(concept_dict)

    mlp=MLP(input_dim=concept_nums, resolution=512//64).to(device)
    unet.set_controlnet(mlp)
    unet=load_weights(unet, mlp_path)

    
    condition=torch.zeros(1,concept_nums, device=device)
    condition[:,concept_dict['antisexual']]=1


    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    torch_device = device

    # 读取csv文件 
    df = pd.read_csv(prompts_path)
    os.makedirs(save_path, exist_ok=True)

    print(f'SelfDiscover 任务: {evaluation_task}，加载模型：{model_name}, 保存地址: {save_path}')

    for _, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.seed
        if evaluation_task == 'i2p':
            case_number = row.case_number
        elif evaluation_task == 'coco':
            case_number = row.image_id
        if case_number<from_case:
            continue

        height = image_size                        # default height of Stable Diffusion
        width = image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = guidance_scale            # Scale for classifier-free guidance

        generator = torch.manual_seed(seed)        # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, controlnet_cond_guidance=condition,
                    controlnet_interpolator=None).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            im.save(f"{save_path}/{case_number}_{num}.png")

    

