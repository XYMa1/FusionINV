"""
低光照图像增强模块
实现基于 Retinex 理论的结构保留增强
"""

import numpy as np
import cv2
from typing import Tuple
from skimage import exposure


def retinex_decompose(image: np.ndarray, sigma: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retinex 分解：将图像分解为反射分量和光照分量

    Args:
        image: 输入图像 [H, W, 3], uint8, RGB
        sigma: 高斯模糊的标准差，控制光照分量的平滑度

    Returns:
        R: 反射分量（物体固有颜色）[H, W, 3], float32, [0, 1]
        L: 光照分量（亮度）[H, W, 3], float32, [0, 1]
    """
    # 转换到 float32 并归一化
    img_float = image.astype(np.float32) / 255.0

    # 避免 log(0)
    img_float = np.maximum(img_float, 1e-6)

    # 估计光照分量（用高斯模糊近似）
    L = cv2.GaussianBlur(img_float, (0, 0), sigma)
    L = np.maximum(L, 1e-6)

    # 计算反射分量 R = I / L
    R = img_float / L
    R = np.clip(R, 0, 1)

    return R, L


def enhance_illumination(L: np.ndarray, gamma: float = 0.4) -> np.ndarray:
    """
    增强光照分量

    Args: 
        L: 光照分量 [H, W, 3], float32, [0, 1]
        gamma:  Gamma 校正系数，越小提亮越强（0.4 用于极暗场景）

    Returns: 
        L_enhanced: 增强后的光照 [H, W, 3], float32, [0, 1]
    """
    L_enhanced = np.power(L, gamma)
    return L_enhanced


def apply_clahe(image: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """
    应用自适应直方图均衡化（CLAHE）

    Args:
        image: 输入图像 [H, W, 3], float32, [0, 1]
        clip_limit: 对比度限制，防止过度增强

    Returns:
        enhanced: 均衡化后的图像 [H, W, 3], float32, [0, 1]
    """
    # 转换到 uint8
    img_uint8 = (image * 255).astype(np.uint8)

    # 转换到 LAB 空间（只对 L 通道做均衡）
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # 合并回去
    lab_clahe = cv2.merge([l_clahe, a, b])
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe.astype(np.float32) / 255.0


def enhance_low_light(image: np.ndarray,
                      exposure: float = None,
                      sigma: int = 15) -> np.ndarray:
    """
    完整的低光增强流程（主函数）

    Args: 
        image: 输入图像 [H, W, 3], uint8, RGB
        exposure: 曝光度（0-1），如果为 None 则自动计算
        sigma:  Retinex 分解的高斯核大小

    Returns: 
        enhanced: 增强后的图像 [H, W, 3], uint8, RGB
    """
    # 1. 计算曝光度（如果未提供）
    if exposure is None:
        from utils.exposure_metrics import compute_exposure
        exposure = compute_exposure(image)

    # 2. 根据曝光度选择 Gamma 值
    if exposure < 0.2:
        gamma = 0.4  # 极暗
    elif exposure < 0.4:
        gamma = 0.5  # 弱光
    else:
        gamma = 0.6  # 正常偏暗

    print(f"  [增强] 曝光度={exposure:.3f}, Gamma={gamma}")

    # 3. Retinex 分解
    R, L = retinex_decompose(image, sigma=sigma)

    # 4. 增强光照分量
    L_enhanced = enhance_illumination(L, gamma=gamma)

    # 5. 重组图像
    I_enhanced = R * L_enhanced
    I_enhanced = np.clip(I_enhanced, 0, 1)

    # 6.  CLAHE 自适应均衡
    I_final = apply_clahe(I_enhanced, clip_limit=0.03)

    # 7. 转回 uint8
    result = (I_final * 255).astype(np.uint8)

    return result


# ========== 测试代码 ==========
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image

    # 测试用例
    test_image_path = "data/LLVIP/vi/010001.jpg"  # 替换为你的测试图像

    try:
        img = np.array(Image.open(test_image_path).convert('RGB'))
        print(f"✅ 加载测试图像：{img.shape}")

        # 执行增强
        enhanced = enhance_low_light(img)

        # 可视化对比
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original (Low-Light)")
        axes[0].axis('off')

        axes[1].imshow(enhanced)
        axes[1].set_title("Enhanced (Retinex)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig("test_enhancement.png", dpi=150, bbox_inches='tight')
        print("✅ 测试完成，结果保存到 test_enhancement.png")

    except FileNotFoundError:
        print("❌ 测试图像未找到，请修改 test_image_path")
