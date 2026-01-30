'''
Description: 
Author: Zhou Tianyi
Date: 2025-01-15 02:45:11
LastEditTime: 2025-04-19 13:58:09
LastEditors:  
'''
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tuneavideo.util import save_videos_grid,ddim_inversion
import torch
from models.train_semantic_predictor import CLIP
import numpy as np
from einops import rearrange
from sklearn import preprocessing
import random
import math
#pretrained_eeg_encoder_path = '/home/v-xuanhaoliu/EEG2Video/Tune-A-Video/tuneavideo/models/eeg2text_40_eeg.pt'
model = None
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
)
torch.cuda.set_device(7)
def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')
# seed_everything(114514)

eeg = torch.load('text_embeds/MindCine/block7_10classes_114514-sub7.pt',map_location='cpu')


negative = eeg.mean(dim=0)

pretrained_model_path = "huggingface/stable-diffusion-v1-5"
my_model_path = "Tune-A-Video/outputs/10_classes_500_epoch"

unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path ,unet=unet, torch_dtype=torch.float16).to("cuda")
inference_scheduler = DDIMScheduler.from_pretrained("huggingface/stable-diffusion-v1-5/scheduler")


ddim_inv_scheduler = inference_scheduler
ddim_inv_scheduler.set_timesteps(50)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()


weight_dtype = torch.float16
latents = np.load('latents/latent_out_block7_10_classes.npy')
latents = torch.from_numpy(latents).half()
latents = rearrange(latents, 'a b c d e -> a c b d e')
latents = latents.to('cuda')

ddim_inv_latent =None 
for i in range(len(latents)):
    temp_latent = latents[i].unsqueeze(0)
    temp = ddim_inversion(unet, ddim_inv_scheduler, video_latent=temp_latent, num_inv_steps=50, prompt='')[-1].to(weight_dtype)
    
    if ddim_inv_latent is None:
        ddim_inv_latent = temp
    else:
        ddim_inv_latent = torch.cat((ddim_inv_latent, temp), dim=0)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
latents = ddim_inv_latent


noise_scheduler = DDPMScheduler.from_pretrained("huggingface/stable-diffusion-v1-5/scheduler")
noise = torch.randn_like(latents)
timesteps = torch.tensor([100]).long()
latents_add_noise = noise_scheduler.add_noise(latents, noise, timesteps)
latents_add_noise = latents
# del latents
# print(latents.shape)

woSeq2Seq = False
woDANA = False 
woSemantic = False
noise_eeg = torch.zeros_like(eeg)
for i in range(len(eeg)):
    if woSeq2Seq:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative, latents=None, video_length=6, height=288, width=512, num_inference_steps=100, guidance_scale=12.5).videos
        savename = f'Tune-A-Video/10classes_woSeq2Seq/'
        import os
        os.makedirs(savename, exist_ok=True)
    elif woSemantic:
        # 要改pipeline
        video = pipe(model, noise_eeg[i:i+1,...], negative_eeg=negative,latents=latents_add_noise[i:i+1,...], video_length=6, height=256, width=256, num_inference_steps=100, guidance_scale=15.5).videos
        savename = f'Tune-A-Video/10classes_woSemantic/'
        import os
        os.makedirs(savename, exist_ok=True)
    else:
        video = pipe(model, eeg[i:i+1,...],negative_eeg=negative,latents=latents_add_noise[i:i+1,...], video_length=6, height=288, width=512 , num_inference_steps=100, guidance_scale=7.5).videos
        savename = f'Tune-A-Video/10classes_fullmodel/'
        import os
        os.makedirs(savename, exist_ok=True)
    # print(f"{savename}/{i}.gif")
    save_videos_grid(video, f"{savename}/{i}.gif")
    # exit()

 