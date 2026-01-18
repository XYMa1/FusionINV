"""
混合融合策略：传统融合 + SD融合的软加权
针对低光照场景优化 - 简化版
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional
import math
import cv2


def compute_confidence_weight(E_vi: float,
                              center: float = 0.25,
                              smooth: float = 0.1) -> float:
    """
    计算SD融合的置信度权重（基于曝光度）

    Args:
        E_vi: 曝光度 [0, 1]
        center:   Sigmoid中心点（默认0.25）
        smooth:  平滑系数（默认0.1）

    Returns:
        alpha: SD融合的权重 [0, 1]
    """
    # Sigmoid函数
    alpha = 1.0 / (1.0 + math.exp(-(E_vi - center) / smooth))

    # 裁剪到 [0, 1]
    alpha = max(0.0, min(1.0, alpha))

    return alpha


def traditional_fusion(vis_image: Image.Image,
                       ir_image: Image.Image,
                       w_vi: float = 0.6,
                       w_ir: float = 0.6,
                       gamma: float = 0.65,
                       saturation_boost: float = 1.2) -> Image.Image:
    """
    传统加权融合（简化版，作为SD的输入基础）

    策略：
      1. 简单的HSV亮度融合
      2. 温和的饱和度增强
      3. 避免过度处理（细节交给SD）

    Args:
        vis_image: 可见光图像
        ir_image:  红外图像
        w_vi:  VI权重
        w_ir: IR权重
        gamma:  Gamma校正系数
        saturation_boost:  饱和度增强系数

    Returns:
        融合图像
    """
    # 转换为numpy数组
    vis_np = np.array(vis_image).astype(np.uint8)
    ir_np = np.array(ir_image).astype(np.uint8)

    # IR转灰度
    if len(ir_np.shape) == 3:
        ir_gray = cv2.cvtColor(ir_np, cv2.COLOR_RGB2GRAY)
    else:
        ir_gray = ir_np

    # ========== 简化的HSV融合 ==========

    # 1. VI转HSV
    vis_hsv = cv2.cvtColor(vis_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    vis_h = vis_hsv[:, :, 0]  # 色调（保留）
    vis_s = vis_hsv[:, :, 1]  # 饱和度（轻微增强）
    vis_v = vis_hsv[:, :, 2] / 255.0  # 亮度（需要增强）

    # 2. IR归一化
    ir_gray_norm = ir_gray.astype(np.float32) / 255.0

    # 3. 简单的亮度融合
    fused_v = w_vi * vis_v + w_ir * ir_gray_norm

    # 4. Gamma校正
    fused_v = np.power(np.clip(fused_v, 0, 1), gamma)

    # 5. 温和的饱和度增强
    vis_s_enhanced = np.clip(vis_s * saturation_boost, 0, 255)

    # 6. 重组HSV
    fused_hsv = np.stack([
        vis_h,
        vis_s_enhanced,
        (fused_v * 255).astype(np.float32)
    ], axis=-1).astype(np.uint8)

    # 7. 转回RGB
    fused_rgb = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2RGB)

    # ===================================

    return Image.fromarray(fused_rgb)


def blend_images(img1: Image.Image,
                 img2: Image.Image,
                 alpha: float) -> Image.Image:
    """
    线性混合两张图像

    Args:
        img1: 图像1（SD融合结果）
        img2: 图像2（传统融合结果）
        alpha: img1的权重

    Returns:
        混合图像
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    # 线性混合
    blended = alpha * arr1 + (1 - alpha) * arr2

    # 转换回uint8
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_uint8)


def adaptive_config_for_exposure(E_vi: float, base_cfg) -> dict:
    """
    根据曝光度自适应调整SD融合参数

    Args:
        E_vi: 曝光度
        base_cfg: 基础配置对象

    Returns:
        调整后的参数字典
    """
    params = {}

    # ========== 修改：降低阈值，强制运行SD ==========
    if E_vi < 0.10:  # 只有极极暗才跳过SD
        params['run_sd'] = False
        params['skip_steps'] = 70
        params['swap_guidance_scale'] = 1.05
    # ==============================================

    elif E_vi < 0.25:
        # 很暗：极端保留
        params['run_sd'] = True
        params['skip_steps'] = 70
        params['swap_guidance_scale'] = 1.05
        params['cfg_scale_src'] = 1.0

    elif E_vi < 0.4:
        # 弱光：保守融合
        params['run_sd'] = True
        params['skip_steps'] = 40
        params['swap_guidance_scale'] = 1.5
        params['cfg_scale_src'] = 1.0

    else:
        # 正常：标准融合
        params['run_sd'] = True
        params['skip_steps'] = 25
        params['swap_guidance_scale'] = 3.5
        params['cfg_scale_src'] = 3.5

    return params


def print_fusion_strategy(E_vi: float, alpha: float, params: dict):
    """
    打印融合策略信息
    """
    print("\n" + "=" * 70)
    print("🎯 混合融合策略（低光照优化版 - 启用SD）")
    print("=" * 70)
    print(f"  📊 场景曝光度: E_vi = {E_vi:.4f}")
    print(f"  ⚖️  置信度权重: α = {alpha:.3f}")
    print(f"      → SD融合权重:       {alpha * 100:.1f}%")
    print(f"      → 传统融合权重:  {(1 - alpha) * 100:.1f}%")
    print(f"  ⚙️  SD参数:")
    print(f"      → 是否运行SD:     {params['run_sd']}")
    if params['run_sd']:
        print(f"      → skip_steps:     {params['skip_steps']}")
        print(f"      → guidance_scale: {params['swap_guidance_scale']}")
        print(f"      → cfg_scale_src:  {params.get('cfg_scale_src', 'N/A')}")
    print(f"  🎨 传统融合策略:")
    print(f"      → 简化HSV融合（避免过度处理）")
    print(f"      → Gamma校正 (γ=0.65)")
    print(f"      → 温和饱和度增强 (×1.2)")
    print("=" * 70 + "\n")
