from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diffusers.training_utils import set_seed

from AllinVIS import AllinVISModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion_vis import invert
from utils.ddpm_inversion_inf2vis import invertinf
# 在现有 import 后添加
from utils.low_light_enhance import enhance_low_light
from utils.exposure_metrics import compute_exposure


def load_latents_or_invert_images(model: AllinVISModel, cfg: RunConfig):
    if cfg.load_latents and cfg.vis_latent_save_path.exists() and cfg.ir_latent_save_path.exists():
        print("Loading existing latents...")
        latents_vis, latents_ir = load_latents(cfg.vis_latent_save_path, cfg.ir_latent_save_path)
        noise_vis, noise_ir = load_noise(cfg.vis_latent_save_path, cfg.ir_latent_save_path)
        print("Done.")
    else:
        print("Inverting images...")
        vis_image, ir_image = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_vis, latents_ir, noise_vis, noise_ir = invert_images(vis_image=vis_image,
                                                                             ir_image=ir_image,
                                                                             sd_model=model.pipe,
                                                                             cfg=cfg)
        model.enable_edit = True
        print("Done.")
    return latents_vis, latents_ir, noise_vis, noise_ir


def load_latents(vis_latent_save_path: Path, ir_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_vis = torch.load(vis_latent_save_path)
    latents_ir = torch.load(ir_latent_save_path)
    if type(latents_ir) == list:
        latents_vis = [l.to("cuda") for l in latents_vis]
        latents_ir = [l.to("cuda") for l in latents_ir]
    else:
        latents_vis = latents_vis.to("cuda")
        latents_ir = latents_ir.to("cuda")
    return latents_vis, latents_ir


def load_noise(vis_latent_save_path: Path, ir_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_vis = torch.load(vis_latent_save_path.parent / (vis_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_ir = torch.load(ir_latent_save_path.parent / (ir_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_vis = latents_vis.to("cuda")
    latents_ir = latents_ir.to("cuda")
    return latents_vis, latents_ir


def invert_images(sd_model, vis_image: Image.Image, ir_image: Image.Image, cfg: RunConfig):
    """
    改进的图像反演流程：集成低光增强 + 动态Prompt
    """
    print("\n" + "=" * 70)
    print("🔄 开始图像反演 (DDPM Inversion)")
    print("=" * 70)

    # ========== 可见光路：增强 + 反演 ==========
    print("\n[阶段 1] 可见光图像预处理")

    # 1. 计算原始曝光度
    vis_np = np.array(vis_image)
    from utils.exposure_metrics import compute_exposure, generate_dynamic_prompt
    E_vi = compute_exposure(vis_np)
    print(f"  📊 原始曝光度:  E_vi = {E_vi:. 4f}")

    # 2. 低光增强（如果需要）
    if E_vi < 0.15:
        # 极暗场景：跳过增强以保留细节
        print(f"  ⚠️  场景极暗 (E_vi={E_vi:.3f})，跳过增强以保留细节")
        vis_enhanced_np = vis_np
    elif E_vi < 0.6:
        # 弱光场景：执行增强
        print(f"  🔧 检测到低光场景，执行 Retinex 增强...")
        from utils.low_light_enhance import enhance_low_light
        vis_enhanced_np = enhance_low_light(vis_np, exposure=E_vi)
        vis_enhanced = Image.fromarray(vis_enhanced_np)

        # 保存增强后的图像（用于调试）
        if cfg.output_path:
            vis_enhanced.save(cfg.output_path / "vi_enhanced.png")
            print(f"  ✅ 增强图像已保存:  vi_enhanced.png")
    else:
        # 光照充足：跳过增强
        print(f"  ✅ 光照充足，跳过增强")
        vis_enhanced_np = vis_np

    # 3. 生成动态Prompt（改进版 - 支持Negative）
    positive_prompt, negative_prompt = generate_dynamic_prompt(E_vi)

    if positive_prompt:
        print(f"  📝 动态Prompt (Positive): \"{positive_prompt}\"")
        print(f"  📝 动态Prompt (Negative): \"{negative_prompt}\"")
    else:
        print(f"  📝 使用空Prompt（场景光照正常）")

    # 4. 转换为 tensor
    input_vis = torch.from_numpy(vis_enhanced_np).float() / 127.5 - 1.0
    input_vis = input_vis.permute(2, 0, 1).unsqueeze(0).to('cuda')

    # 5. DDPM Inversion（VI路）
    print(f"  🔄 执行 DDPM Inversion (VI路)...")
    set_seed(cfg.seed)

    # 【关键修改】使用动态Prompt和更高的CFG
    from utils.ddpm_inversion_vis import invert
    zs_vis, latents_vis, directions = invert(
        x0=input_vis,
        pipe=sd_model,
        prompt_src=positive_prompt,  # 使用动态Prompt
        num_diffusion_steps=cfg.num_timesteps,
        cfg_scale_src=5.5 if E_vi < 0.3 else 3.5,  # 低光场景提高CFG
        eta=1,
        use_dynamic_prompt=False,  # 已经手动生成了
        exposure=E_vi
    )

    # ========== 红外路：标准反演 ==========
    print("\n[阶段 2] 红外图像反演")
    input_ir = torch.from_numpy(np.array(ir_image)).float() / 127.5 - 1.0
    input_ir = input_ir.permute(2, 0, 1).unsqueeze(0).to('cuda')

    set_seed(cfg.seed)
    from utils.ddpm_inversion_inf2vis import invertinf
    direction_step_size = float(cfg.direction_step_size)

    print(f"  🔄 执行 DDPM Inversion (IR路)...")
    zs_ir, latents_ir = invertinf(
        x0=input_ir,
        vis_direction=directions,
        direction_step_size=direction_step_size,
        pipe=sd_model,
        num_diffusion_steps=cfg.num_timesteps,
        cfg_scale_src=3.5
    )

    # ========== 保存 latents（可选）==========
    if not cfg.load_latents:
        print(f"\n💾 保存 Latents 到 {cfg.latents_path}")
        torch.save(latents_vis, cfg.latents_path / f"{cfg.vis_image_path.stem}_vis.pt")
        torch.save(latents_ir, cfg.latents_path / f"{cfg.ir_image_path.stem}_ir.pt")
        torch.save(zs_vis, cfg.latents_path / f"{cfg.vis_image_path.stem}_vis_ddpm_noise.pt")
        torch.save(zs_ir, cfg.latents_path / f"{cfg.ir_image_path.stem}_ir_ddpm_noise. pt")

    # ========== 保存曝光度信息（供后续使用）==========
    cfg.E_vi = E_vi

    # 转换格式
    if isinstance(zs_ir, list):
        zs_ir = torch.stack(zs_ir)
    if isinstance(latents_ir, list):
        latents_ir = torch.stack(latents_ir)

    print("\n" + "=" * 70)
    print(f"✅ 反演完成 (E_vi={E_vi:.4f})")
    print("=" * 70 + "\n")

    return latents_vis, latents_ir, zs_vis, zs_ir


def get_init_latents_and_noises(model: AllinVISModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_ir.dim() == 4 and model.latents_vis.dim() == 4 and model.latents_vis.shape[0] > 1:
        model.latents_ir = model.latents_ir[cfg.skip_steps]
        model.latents_vis = model.latents_vis[cfg.skip_steps]
    init_latents = torch.stack([model.latents_vis, model.latents_vis, model.latents_ir])
    zs_fusion = model.zs_vis.clone()
    init_zs = [ zs_fusion[cfg.skip_steps:], model.zs_vis[cfg.skip_steps:], model.zs_ir[cfg.skip_steps:]]
    return init_latents, init_zs
