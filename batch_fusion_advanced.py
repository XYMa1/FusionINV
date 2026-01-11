"""
LIT-Fusion 批量融合脚本（多数据集支持）
支持 LLVIP、TNO、MSRS 三个数据集的批量处理
"""

import sys

sys.path.append(".")

from pathlib import Path
import torch
from tqdm import tqdm
from config import RunConfig
from fusioninv import run, set_seed


def batch_fusion_dataset(
        dataset_name: str,
        vi_dir: Path,
        ir_dir: Path,
        output_dir: Path,
        num_images: int = None,
        num_timesteps: int = 100,
        load_latents: bool = True,  # 首次False，后续True加速
):
    """
    批量融合单个数据集

    Args:
        dataset_name:  数据集名称（如 "LLVIP", "TNO", "MSRS"）
        vi_dir: 可见光图像文件夹
        ir_dir: 红外图像文件夹
        output_dir:  输出根目录
        num_images:  处理图像数量（None=全部）
        num_timesteps: 去噪步数
        load_latents: 是否加载已保存的latents（加速）
    """
    print("\n" + "=" * 70)
    print(f"🚀 开始处理数据集: {dataset_name}")
    print("=" * 70)
    print(f"  可见光文件夹: {vi_dir}")
    print(f"  红外文件夹:     {ir_dir}")
    print(f"  输出目录:     {output_dir / dataset_name}")
    print(f"  去噪步数:    {num_timesteps}")
    print(f"  加载latents: {load_latents}")
    print("=" * 70 + "\n")

    # 获取图像列表（支持 . jpg, .png, .bmp）
    vi_images = sorted(
        list(vi_dir.glob("*.jpg")) +
        list(vi_dir.glob("*.png")) +
        list(vi_dir.glob("*.bmp"))
    )
    ir_images = sorted(
        list(ir_dir.glob("*.jpg")) +
        list(ir_dir.glob("*.png")) +
        list(ir_dir.glob("*.bmp"))
    )

    # 验证图像对数量
    if len(vi_images) != len(ir_images):
        print(f"⚠️  警告:  VI图像数({len(vi_images)}) != IR图像数({len(ir_images)})")
        num_pairs = min(len(vi_images), len(ir_images))
    else:
        num_pairs = len(vi_images)

    # 限制处理数量
    if num_images is not None:
        num_pairs = min(num_pairs, num_images)

    print(f"📊 共找到 {num_pairs} 对图像\n")

    # 批量处理
    success_count = 0
    failed_list = []

    for idx in tqdm(range(num_pairs), desc=f"处理 {dataset_name}"):
        try:
            vi_path = vi_images[idx]
            ir_path = ir_images[idx]

            # 验证文件名是否匹配
            if vi_path.stem != ir_path.stem:
                print(f"\n⚠️  警告: 文件名不匹配 - VI: {vi_path.name}, IR: {ir_path.name}")

            # 创建配置
            cfg = RunConfig(
                vis_image_path=vi_path,
                ir_image_path=ir_path,
                domain_name=dataset_name,
                output_path=output_dir,
                num_timesteps=num_timesteps,
                load_latents=load_latents,
                skip_steps=32,
                seed=42,
            )

            # 运行融合
            set_seed(cfg.seed)
            run(cfg)

            success_count += 1

        except Exception as e:
            print(f"\n❌ 处理失败 [{idx + 1}]: {vi_path.name}")
            print(f"   错误: {e}")
            failed_list.append((idx + 1, vi_path.name, str(e)))
            continue

    # 统计结果
    print("\n" + "=" * 70)
    print(f"✅ {dataset_name} 处理完成")
    print("=" * 70)
    print(f"  成功:  {success_count}/{num_pairs}")
    print(f"  失败: {len(failed_list)}")
    if failed_list:
        print("\n失败列表:")
        for idx, name, error in failed_list:
            print(f"  [{idx}] {name}: {error}")
    print("=" * 70 + "\n")

    return success_count, failed_list


def main():
    """
    主函数：批量处理三个数据集
    """
    # ========== 配置数据集路径 ==========
    base_dir = Path("D:\mxy\FusionINV-main")  # ← 修改为你的项目路径

    datasets = [
        {
            "name": "LLVIP",
            "vi_dir": base_dir / "data/LLVIP/vi",
            "ir_dir": base_dir / "data/LLVIP/ir",
            "num_images": None,  # None=处理全部，或指定数量如 10
        },
        {
            "name": "TNO",
            "vi_dir": base_dir / "data/TNO/vi",
            "ir_dir": base_dir / "data/TNO/ir",
            "num_images": None,
        },
        {
            "name": "MSRS",
            "vi_dir": base_dir / "data/MSRS/vi",
            "ir_dir": base_dir / "data/MSRS/ir",
            "num_images": None,
        },
    ]

    output_dir = base_dir / "output"

    # ========== 全局配置 ==========
    num_timesteps = 100  # 去噪步数（推荐100）
    load_latents = False  # 首次False，之后True加速（跳过inversion）
    # ============================

    print("\n" + "🔥" * 35)
    print("LIT-Fusion 批量处理 - 多数据集模式")
    print("🔥" * 35)
    print(f"\n将处理以下数据集:")
    for ds in datasets:
        print(f"  • {ds['name']}:  {ds['vi_dir']}")
    print(f"\n输出目录: {output_dir}")
    print(f"去噪步数: {num_timesteps}")
    print(f"加载latents: {load_latents}")

    input("\n按 Enter 开始处理...")

    # ========== 批量处理 ==========
    total_success = 0
    total_failed = 0

    for ds_config in datasets:
        success, failed = batch_fusion_dataset(
            dataset_name=ds_config["name"],
            vi_dir=ds_config["vi_dir"],
            ir_dir=ds_config["ir_dir"],
            output_dir=output_dir,
            num_images=ds_config["num_images"],
            num_timesteps=num_timesteps,
            load_latents=load_latents,
        )
        total_success += success
        total_failed += len(failed)

    # ========== 最终统计 ==========
    print("\n" + "🎉" * 35)
    print("所有数据集处理完成！")
    print("🎉" * 35)
    print(f"\n总成功: {total_success}")
    print(f"总失败: {total_failed}")
    print(f"\n结果保存在: {output_dir}")
    print("\n文件夹结构:")
    print("  output/")
    print("    ├── LLVIP/")
    print("    │   ├── 1/  (第1对图片)")
    print("    │   ├── 2/  (第2对图片)")
    print("    │   └── ...")
    print("    ├── TNO/")
    print("    │   ├── 1/")
    print("    │   └── ...")
    print("    └── MSRS/")
    print("        ├── 1/")
    print("        └── ...")


if __name__ == "__main__":
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用:  {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA 不可用，将使用 CPU（速度很慢）")

    main()
