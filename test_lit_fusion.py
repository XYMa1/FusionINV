"""
LIT-Fusion 端到端测试脚本
测试完整的低光照融合流程
"""

import sys

sys.path.append(".")

from pathlib import Path
import torch
import numpy as np
from PIL import Image

from config import RunConfig
from AllinVIS import AllinVISModel
from utils.latent_utils import load_latents_or_invert_images
from utils.exposure_metrics import compute_exposure, get_illumination_level


def create_test_images():
    """创建测试用的低光图像（如果没有真实数据）"""
    # 创建一个模拟的低光可见光图像
    vi_image = np.ones((512, 512, 3), dtype=np.uint8) * 30  # 很暗
    vi_image[100:200, 100:200] = [50, 50, 60]  # 添加一些细节

    # 创建一个模拟的红外图像
    ir_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    ir_image[150:250, 150:250] = 200  # 热目标

    # 保存
    output_dir = Path("data/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(vi_image).save(output_dir / "test_vi_lowlight.png")
    Image.fromarray(ir_image).save(output_dir / "test_ir. png")

    print(f"✅ 测试图像已创建: {output_dir}")
    return output_dir / "test_vi_lowlight. png", output_dir / "test_ir.png"


def test_exposure_calculation():
    """测试曝光度计算"""
    print("\n" + "=" * 60)
    print("测试 1: 曝光度计算")
    print("=" * 60)

    # 创建不同亮度的测试图像
    test_cases = [
        ("极暗", np.ones((100, 100, 3), dtype=np.uint8) * 20),
        ("很暗", np.ones((100, 100, 3), dtype=np.uint8) * 50),
        ("弱光", np.ones((100, 100, 3), dtype=np.uint8) * 100),
        ("正常", np.ones((100, 100, 3), dtype=np.uint8) * 150),
    ]

    for name, img in test_cases:
        E_vi = compute_exposure(img)
        level = get_illumination_level(E_vi)
        print(f"  {name}: E_vi={E_vi:.4f}, 等级={level}")

    print("✅ 曝光度计算测试通过")


def test_model_initialization():
    """测试模型初始化"""
    print("\n" + "=" * 60)
    print("测试 2: 模型初始化")
    print("=" * 60)

    vi_path, ir_path = create_test_images()

    cfg = RunConfig(
        vis_image_path=vi_path,
        ir_image_path=ir_path,
        domain_name="test_lowlight",
        num_timesteps=50,
        use_masked_adain=False,
        load_latents=False,  # 强制重新反演
        skip_steps=16  # 减少步数加快测试
    )

    try:
        model = AllinVISModel(cfg)
        print(f"  ✅ 模型创建成功")
        print(f"  - total_steps: {model.total_steps}")
        print(f"  - E_vi (初始): {model.E_vi}")
        print(f"  - enable_edit: {model.enable_edit}")

        # 测试权重计算
        print("\n  测试权重计算:")
        for step in [0, 25, 45]:
            current_t = model.total_steps - step
            w1, w2, w3 = model.compute_adaptive_weights(current_t)
            print(f"    step={step}, t={current_t}:  w_ir={w1:.3f}, w_vi={w2:.3f}, w_txt={w3:.3f}")

        print("✅ 模型初始化测试通过")
        return model, cfg

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_inversion(model, cfg):
    """测试反演流程"""
    print("\n" + "=" * 60)
    print("测试 3: 图像反演（含增强）")
    print("=" * 60)

    try:
        latents_vis, latents_ir, noise_vis, noise_ir = load_latents_or_invert_images(
            model=model,
            cfg=cfg
        )

        print(f"  ✅ 反演完成")
        print(f"  - latents_vis shape: {latents_vis.shape}")
        print(f"  - latents_ir shape: {latents_ir.shape}")
        print(f"  - 曝光度 E_vi: {cfg.E_vi:. 4f}")

        # 更新模型的曝光度
        model.E_vi = cfg.E_vi
        model.set_latents(latents_vis, latents_ir)
        model.set_noise(noise_vis, noise_ir)

        print("✅ 反演测试通过")
        return True

    except Exception as e:
        print(f"❌ 反演失败:  {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion(model, cfg):
    """测试融合流程"""
    print("\n" + "=" * 60)
    print("测试 4: 端到端融合")
    print("=" * 60)

    try:
        from utils.latent_utils import get_init_latents_and_noises
        from config import Range

        init_latents, init_zs = get_init_latents_and_noises(model=model, cfg=cfg)
        model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
        model.enable_edit = True

        print(f"  开始融合（{cfg.num_timesteps - cfg.skip_steps} 步）...")

        visir_prompts = [cfg.prompt, "", ""]

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
            cross_image_attention_range=Range(start=1, end=90),
        ).images

        # 保存结果
        images[0].save(cfg.output_path / "test_fusion. png")
        images[1].save(cfg.output_path / "test_vis.png")
        images[2].save(cfg.output_path / "test_ir.png")

        print(f"  ✅ 融合完成")
        print(f"  - 结果保存至: {cfg.output_path}")
        print(f"  - 融合图像:  test_fusion.png")

        print("✅ 融合测试通过")
        return True

    except Exception as e:
        print(f"❌ 融合失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "🚀" * 30)
    print("LIT-Fusion 端到端测试")
    print("🚀" * 30)

    # 测试 1: 曝光度计算
    test_exposure_calculation()

    # 测试 2: 模型初始化
    model, cfg = test_model_initialization()
    if model is None:
        print("\n❌ 测试中止：模型初始化失败")
        return

    # 测试 3: 反演
    if not test_inversion(model, cfg):
        print("\n❌ 测试中止：反演失败")
        return

    # 测试 4: 融合
    if not test_fusion(model, cfg):
        print("\n❌ 测试中止：融合失败")
        return

    # 全部通过
    print("\n" + "🎉" * 30)
    print("所有测试通过！LIT-Fusion 已成功部署！")
    print("🎉" * 30)
    print("\n下一步：")
    print("  1. 使用真实的低光图像测试")
    print("  2. 运行批量测试脚本")
    print("  3. 调整参数优化结果")


if __name__ == "__main__":
    # 设置设备
    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA 不可用，将使用 CPU（速度较慢）")

    main()
