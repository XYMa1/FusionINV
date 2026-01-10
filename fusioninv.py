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


# ========== 新增：无参数运行入口 ==========
def main_with_defaults():
    """
    使用默认配置运行（方便PyCharm直接运行）
    """
    from pathlib import Path

    # 设置默认参数
    cfg = RunConfig(
        vis_image_path=Path("data/LLVIP/vi/1.jpg"),
        ir_image_path=Path("data/LLVIP/ir/1.jpg"),
        domain_name="LLVIP",
        num_timesteps=100,
        load_latents=False,  # 首次运行设为False，之后改为True加速
        skip_steps=32,
        seed=42,
    )

    print("=" * 60)
    print("使用默认配置运行 LIT-Fusion")
    print("=" * 60)
    print(f"  可见光图像: {cfg.vis_image_path}")
    print(f"  红外图像:    {cfg.ir_image_path}")
    print(f"  去噪步数:   {cfg.num_timesteps}")
    print(f"  域名称:     {cfg.domain_name}")
    print("=" * 60)

    run(cfg)


# =========================================


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
        print(f"\n[LIT-Fusion] 曝光度已设置:  E_vi={model.E_vi:.4f}")
    # ==========================================

    print("Running visible infrared fusion...")
    images = run_infraredvisiblefusion(model=model, cfg=cfg)
    print("Done.")
    return images


def run_infraredvisiblefusion(model: AllinVISModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)

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
    # ========== 修改：使用输入文件名 ==========
    base_name = cfg.ir_image_path.stem  # 使用IR图像的文件名（如 "1"）

    # Save images
    images[0].save(cfg.output_path / f"{base_name}.png")  # 融合结果：1.png
    images[1].save(cfg.output_path / f"out_vis_{base_name}.png")  # out_vis_1.png
    images[2].save(cfg.output_path / f"out_ir_{base_name}.png")  # out_ir_1.png

    print(f"\n✅ 融合完成，结果已保存:")
    print(f"  融合:   {cfg.output_path}/{base_name}.png")
    print(f"  可见光:  {cfg.output_path}/out_vis_{base_name}.png")
    print(f"  红外:   {cfg.output_path}/out_ir_{base_name}.png")
    # =========================================

    return images


if __name__ == '__main__':
    import sys

    # 如果没有命令行参数，使用默认配置
    if len(sys.argv) == 1:
        print("⚡ 检测到无参数运行，使用默认配置...")
        main_with_defaults()
    else:
        # 有参数时使用命令行参数
        main()
