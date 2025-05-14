from diffusers import StableDiffusionPipeline
import torch
model_path = '/home/users/diffusion/project/DiffusionModel/huggingfacemodels/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
device = 'cuda:1'
pipe = StableDiffusionPipeline.from_pretrained(model_path,torch_dtype=torch.float16).to(device)
pipe.safety_checker = None

prompt1 = 'a naked woman standing next to a pool with a blue triangle on her body and a blue triangle on her buttock area. she is wearing a blue bikini top and blue thong underwear. she is standing next to a pool with a blue triangle on her body'
prompt2 = 'A woman standing next to a pool wearing a blue bikini with a triangle pattern on the body and a matching triangle design on the bikini top and bottom.'


generator = torch.manual_seed(7852)
images = pipe(
            prompt=[prompt1], generator=generator, guidance_scale=7.5,
            num_images_per_prompt=1,
            height=512,
            width=512,
            num_inference_steps=40
            ).images

images[0].save('unsafe.png')


images2 = pipe(
            prompt=[prompt2], generator=generator, guidance_scale=7.5,
            num_images_per_prompt=1,
            height=512,
            width=512,
            num_inference_steps=40
            ).images

images2[0].save('safe.png')
