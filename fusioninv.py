import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


sys.path.append(".")
sys.path.append("..")

from AllinVIS import AllinVISModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images


@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    
    model = AllinVISModel(cfg)
    latents_vis, latents_ir, noise_vis, noise_ir = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_vis, latents_ir)
    model.set_noise(noise_vis, noise_ir)
    
    # ========== 新增：传递曝光度到模型 ==========
    if hasattr(cfg, 'E_vi'):
        model.E_vi = cfg.E_vi
        print(f"\n[LIT-Fusion] 曝光度已设置:  E_vi={model.E_vi:. 4f}")
    # ==========================================
    
    print("Running visible infrared fusion...")
    images = run_infraredvisiblefusion(model=model, cfg=cfg)
    print("Done.")
    return images


def run_infraredvisiblefusion(model:  AllinVISModel, cfg:  RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg. num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range. end, cfg.cross_attn_64_range.end)
    
    # ========== 修改：提示词设置 ==========
    visir_prompts = [cfg.prompt] * 3  # 使用配置中的提示词
    visir_prompts[1] = ""  # VI 路径：空提示
    visir_prompts[2] = ""  # IR 路径：空提示
    # visir_prompts[0] 保持原样，用于融合路径
    # ====================================
    
    images = model.pipe(
        prompt=visir_prompts,
        latents=init_latents,
        guidance_scale=cfg.swap_guidance_scale,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    
    # Save images
    images[0].save(cfg.output_path / f"out_fusion---seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_vis---seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"out_ir---seed_{cfg.seed}.png")
    
    return images


if __name__ == '__main__':
    main()
