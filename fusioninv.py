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
sys.path.append(". .")

from AllinVIS import AllinVISModel
from config import RunConfig, Range
from utils import latent_utils
from utils. latent_utils import load_latents_or_invert_images


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
        skip_steps=25,
        seed=42,
    )

    print("=" * 60)
    print("使用默认配置运行 LIT-Fusion")
    print("=" * 60)
    print(f"  可见光图像:  {cfg.vis_image_path}")
    print(f"  红外图像:     {cfg.ir_image_path}")
    print(f"  去噪步数:   {cfg.num_timesteps}")
    print(f"  域名称:     {cfg.domain_name}")
    print("=" * 60)

    run(cfg)


# =========================================


def run(cfg: RunConfig) -> List[Image.Image]:
    """
    混合融合主流程：传统融合 + SD融合的软加权
    """
    import numpy as np
    from utils.exposure_metrics import compute_exposure
    from utils.image_utils import load_images
    from utils.hybrid_fusion import (
        traditional_fusion,
        compute_confidence_weight,
        blend_images,
        adaptive_config_for_exposure,
        print_fusion_strategy
    )

    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)

    # ========== 阶段1：加载图像并计算曝光度 ==========
    print("📥 加载图像...")
    vis_img, ir_img = load_images(cfg=cfg, save_path=cfg.output_path)
    vis_np = np.array(vis_img)
    E_vi = compute_exposure(vis_np)

    # ========== 阶段2：计算置信度权重 ==========
    alpha = compute_confidence_weight(E_vi, center=0.25, smooth=0.1)

    # ========== 阶段3：自适应参数调整 ==========
    params = adaptive_config_for_exposure(E_vi, cfg)
    print_fusion_strategy(E_vi, alpha, params)

    # ========== 阶段4：保守分支（传统融合）==========
    print("🔧 [保守分支] 执行传统加权融合...")
    # 基于饱和度的色彩保留 + 多级亮度增强
    # VI主导融合 + 色彩增强 + 细节保留
    # 简化的传统融合（作为SD的基础）
    I_trad = traditional_fusion(vis_img, ir_img,
                                w_vi=0.6,  # 平衡
                                w_ir=0.6,  # 平衡
                                gamma=0.65,  # 提亮
                                saturation_boost=1.2)  # 温和增强

    # 保存传统融合结果（用于对比）
    base_name = cfg.ir_image_path.stem
    I_trad.save(cfg.output_path / f"{base_name}_traditional.png")
    print(f"   ✅ 传统融合完成，已保存为:  {base_name}_traditional.png")

    # ========== 阶段5：生成分支（SD融合）==========
    if params['run_sd']:
        print(f"🚀 [生成分支] 执行SD融合 (skip={params['skip_steps']}, CFG={params['swap_guidance_scale']})...")

        # 应用自适应参数
        cfg.skip_steps = params['skip_steps']
        cfg.swap_guidance_scale = params['swap_guidance_scale']
        cfg.E_vi = E_vi

        # 运行SD融合
        model = AllinVISModel(cfg)
        latents_vis, latents_ir, noise_vis, noise_ir = load_latents_or_invert_images(model=model, cfg=cfg)
        model.set_latents(latents_vis, latents_ir)
        model.set_noise(noise_vis, noise_ir)

        if hasattr(cfg, 'E_vi'):
            model.E_vi = cfg.E_vi

        images_sd = run_infraredvisiblefusion(model=model, cfg=cfg)
        I_sd = images_sd[0]  # 融合结果

        # 保存SD融合结果（用于对比）
        I_sd.save(cfg.output_path / f"{base_name}_sd.png")
        print(f"   ✅ SD融合完成，已保存为: {base_name}_sd.png")

    else:
        # 极暗场景：跳过SD，节省时间
        print("⚡ [生成分支] 场景极暗，跳过SD融合（节省计算资源）")
        I_sd = I_trad  # 使用传统融合作为替代

    # ========== 阶段6：置信度加权混合 ==========
    print(f"⚖️  [混合] 置信度加权混合 (α={alpha:.3f})...")
    I_final = blend_images(I_sd, I_trad, alpha)

    # ========== 阶段7：保存结果 ==========
    I_final.save(cfg.output_path / f"{base_name}.png")

    # 转换numpy数组到PIL Image（如果需要）
    if isinstance(vis_img, np.ndarray):
        vis_img = Image.fromarray(vis_img)
    if isinstance(ir_img, np.ndarray):
        ir_img = Image.fromarray(ir_img)

    vis_img.save(cfg.output_path / f"out_vis_{base_name}.png")
    ir_img.save(cfg.output_path / f"out_ir_{base_name}.png")

    print("\n" + "=" * 70)
    print("✅ 混合融合完成！")
    print("=" * 70)
    print(f"  📁 输出目录:  {cfg.output_path}")
    print(f"  📄 最终结果:  {base_name}.png (混合)")
    print(f"  📄 对比结果:")
    print(f"      → {base_name}_traditional.png (传统融合，α=0)")
    if params['run_sd']:
        print(f"      → {base_name}_sd.png (SD融合，α=1)")
    print(f"  📊 混合权重: {alpha * 100:.1f}% SD + {(1 - alpha) * 100:.1f}% 传统")
    print("=" * 70 + "\n")

    return [I_final, vis_img, ir_img]


def run_infraredvisiblefusion(model: AllinVISModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe. scheduler.set_timesteps(cfg. num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range. end, cfg.cross_attn_64_range.end)

    # ========== 关键修复：强制使用中性提示词 ==========
    print("  [调试] 使用中性提示词:  'a photo' x3")
    visir_prompts = ["a photo", "a photo", "a photo"]
    # ==============================================

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
    images[1]. save(cfg.output_path / f"out_vis_{base_name}.png")  # out_vis_1.png
    images[2].save(cfg. output_path / f"out_ir_{base_name}.png")  # out_ir_1.png

    print(f"\n✅ 融合完成，结果已保存:")
    print(f"  融合:    {cfg.output_path}/{base_name}.png")
    print(f"  可见光:   {cfg.output_path}/out_vis_{base_name}.png")
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
