"""
曝光度评估模块
计算图像的光照水平，用于自适应权重调度
"""

import numpy as np
from typing import Union
from PIL import Image


def compute_exposure(image: Union[np.ndarray, Image. Image]) -> float:
    """
    计算图像的曝光度指标 E_vi
    
    Args:
        image: 输入图像
               - np.ndarray: [H, W, 3], uint8, RGB
               - PIL.Image: RGB 模式
    
    Returns: 
        E_vi: 曝光度，范围 [0, 1]
              0. 0 = 全黑
              0.2 = 极暗
              0.4 = 弱光
              0.6 = 正常
              1.0 = 过曝
    """
    # 转换为 numpy 数组
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 确保是 RGB 格式
    if image.ndim == 2:  # 灰度图
        I_gray = image
    elif image.shape[2] == 3:  # RGB
        # 使用人眼感知权重计算灰度
        I_gray = 0.299 * image[:, :, 0] + \
                 0.587 * image[:, :, 1] + \
                 0.114 * image[:, :, 2]
    else: 
        raise ValueError(f"不支持的图像格式：{image.shape}")
    
    # 归一化到 [0, 1]
    E_vi = np.mean(I_gray) / 255.0
    
    return float(E_vi)


def get_illumination_level(E_vi: float) -> str:
    """
    根据曝光度返回光照等级描述
    
    Args: 
        E_vi: 曝光度 [0, 1]
    
    Returns:
        level: 光照等级字符串
    """
    if E_vi < 0.15:
        return "极暗 (Extremely Dark)"
    elif E_vi < 0.3:
        return "很暗 (Very Dark)"
    elif E_vi < 0.45:
        return "弱光 (Low Light)"
    elif E_vi < 0.6:
        return "正常偏暗 (Dim)"
    elif E_vi < 0.75:
        return "正常 (Normal)"
    else:
        return "明亮 (Bright)"


def generate_dynamic_prompt(E_vi: float) -> str:
    """
    根据曝光度生成动态的 Inversion 提示词
    
    Args:
        E_vi:  曝光度 [0, 1]
    
    Returns:
        prompt: 用于 DDPM Inversion 的提示词
    """
    if E_vi < 0.2:
        # 极暗：强烈的光照重建
        prompt = "extremely dark scene transformed to bright daylight, clear visibility, natural illumination"
    elif E_vi < 0.4:
        # 弱光：中等光照增强
        prompt = "low light scene with natural illumination, enhanced visibility, daytime lighting"
    elif E_vi < 0.6:
        # 正常偏暗：轻微增强
        prompt = "dimly lit scene with improved clarity, better lighting"
    else:
        # 正常：保持原样
        prompt = ""
    
    return prompt


def compute_exposure_statistics(image: np.ndarray) -> dict:
    """
    计算详细的曝光统计信息（用于调试）
    
    Args: 
        image: 输入图像 [H, W, 3], uint8, RGB
    
    Returns:
        stats: 统计字典
    """
    I_gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, : , 2]
    
    stats = {
        'mean': np.mean(I_gray),
        'std': np.std(I_gray),
        'min': np.min(I_gray),
        'max': np.max(I_gray),
        'median': np. median(I_gray),
        'percentile_10': np.percentile(I_gray, 10),
        'percentile_90': np.percentile(I_gray, 90),
        'E_vi': np.mean(I_gray) / 255.0
    }
    
    return stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # 测试图像列表（如果有的话）
    test_images = [
        "data/LLVIP/vi/010001.jpg",  # 替换为你的测试图像
    ]
    
    for img_path in test_images: 
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # 计算曝光度
            E_vi = compute_exposure(img_np)
            level = get_illumination_level(E_vi)
            prompt = generate_dynamic_prompt(E_vi)
            stats = compute_exposure_statistics(img_np)
            
            # 打印结果
            print(f"\n{'='*60}")
            print(f"图像: {img_path}")
            print(f"曝光度 E_vi: {E_vi:.4f}")
            print(f"光照等级: {level}")
            print(f"动态提示词: {prompt}")
            print(f"详细统计:")
            for key, val in stats.items():
                print(f"  {key}: {val:.2f}")
            
        except FileNotFoundError:
            print(f"❌ 图像未找到: {img_path}")
    
    print("\n✅ 曝光度计算模块测试完成")
