#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FusionINV 批量处理脚本
适用于 LLVIP(50)、MSRS(361)、TNO(25) 数据集
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import json

# 数据集配置
DATASETS = {
    'LLVIP': {
        'ir_dir': 'data/LLVIP/ir',
        'vi_dir': 'data/LLVIP/vi',
        'extensions': ['.jpg', '.JPG', '.jpeg', '. JPEG'],
        'count': 50
    },
    'MSRS': {
        'ir_dir': 'data/MSRS/ir',
        'vi_dir': 'data/MSRS/vi',
        'extensions': ['.png', '.PNG'],
        'count': 361
    },
    'TNO': {
        'ir_dir': 'data/TNO/ir',
        'vi_dir': 'data/TNO/vi',
        'extensions': ['.png', '.PNG'],
        'count': 25
    }
}


def find_matching_file(stem, directory, extensions):
    """在目录中查找匹配的文件（不区分大小写扩展名）"""
    dir_path = Path(directory)
    for ext in extensions:
        candidate = dir_path / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_image_pairs(dataset_name, config):
    """获取数据集中的图像对"""
    ir_dir = Path(config['ir_dir'])
    vi_dir = Path(config['vi_dir'])
    extensions = config['extensions']

    # 检查目录是否存在
    if not ir_dir.exists():
        print(f"❌ 红外文件夹不存在: {ir_dir}")
        return []
    if not vi_dir.exists():
        print(f"❌ 可见光文件夹不存在: {vi_dir}")
        return []

    # 获取所有红外图像
    ir_images = []
    for ext in extensions:
        ir_images.extend(ir_dir.glob(f'*{ext}'))
    ir_images = sorted(ir_images)

    print(f"  找到 {len(ir_images)} 张红外图像")

    # 匹配可见光图像
    pairs = []
    unmatched = []

    for ir_path in ir_images:
        # 查找匹配的可见光图像
        vi_path = find_matching_file(ir_path.stem, vi_dir, extensions)

        if vi_path:
            pairs.append({
                'vi': str(vi_path),
                'ir': str(ir_path),
                'name': ir_path.stem,
                'dataset': dataset_name
            })
        else:
            unmatched.append(ir_path.name)

    print(f"  成功配对: {len(pairs)} 对")
    if unmatched:
        print(f"  ⚠️ 未配对:  {len(unmatched)} 个")
        if len(unmatched) <= 5:
            for name in unmatched:
                print(f"    - {name}")

    return pairs


def check_if_processed(pair_info, output_base):
    """检查图像对是否已经处理过"""
    dataset = pair_info['dataset']
    name = pair_info['name']

    # 可能的输出路径
    output_path = Path(output_base) / dataset / dataset / f"vis={name}---ir={name}" / "out_fusion---seed_41.png"

    return output_path.exists()


def process_single_pair(pair_info, output_base, skip_existing=True):
    """处理单对图像"""
    dataset = pair_info['dataset']
    name = pair_info['name']

    # 检查是否已处理
    if skip_existing and check_if_processed(pair_info, output_base):
        return {'status': 'skip', 'name': name, 'dataset': dataset}

    try:
        output_dir = Path(output_base) / dataset
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建命令
        cmd = [
            'python', 'fusioninv.py',
            '--vis_image_path', pair_info['vi'],
            '--ir_image_path', pair_info['ir'],
            '--domain_name', dataset,
            '--output_path', str(output_dir),
            '--use_masked_adain', 'False',
            '--load_latents', 'True'
        ]

        # 运行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode == 0:
            return {'status': 'success', 'name': name, 'dataset': dataset}
        else:
            return {
                'status': 'failed',
                'name': name,
                'dataset': dataset,
                'error': result.stderr[: 300]
            }

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'name': name, 'dataset': dataset}
    except Exception as e:
        return {'status': 'error', 'name': name, 'dataset': dataset, 'error': str(e)}


def process_dataset(dataset_name, config, output_base, skip_existing=True):
    """处理单个数据集（顺序处理）"""
    print(f"\n{'=' * 70}")
    print(f"📂 处理数据集: {dataset_name}")
    print(f"{'=' * 70}")

    # 获取图像对
    pairs = get_image_pairs(dataset_name, config)

    if len(pairs) == 0:
        print(f"❌ 没有找到图像对，跳过 {dataset_name}")
        return {'success': 0, 'failed': 0, 'skip': 0, 'timeout': 0, 'error': 0}

    # 统计结果
    results = {'success': 0, 'failed': 0, 'skip': 0, 'timeout': 0, 'error': 0}
    failed_items = []

    # 处理每对图像
    for pair in tqdm(pairs, desc=f"处理 {dataset_name}", unit="对"):
        result = process_single_pair(pair, output_base, skip_existing)
        results[result['status']] += 1

        if result['status'] in ['failed', 'error']:
            failed_items.append(result)
            tqdm.write(f"  ❌ 失败: {result['name']}")
            if 'error' in result:
                tqdm.write(f"     错误: {result['error'][: 150]}")
        elif result['status'] == 'timeout':
            failed_items.append(result)
            tqdm.write(f"  ⏱️ 超时: {result['name']}")

    # 打印结果
    print(f"\n📊 {dataset_name} 处理结果:")
    print(f"  ✅ 成功: {results['success']}")
    print(f"  ⏭️ 跳过(已存在): {results['skip']}")
    print(f"  ❌ 失败: {results['failed']}")
    print(f"  ⏱️ 超时: {results['timeout']}")
    print(f"  ❗ 错误: {results['error']}")

    # 保存失败列表
    if failed_items:
        failed_file = Path(output_base) / f"{dataset_name}_failed.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"  📄 失败列表已保存: {failed_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='FusionINV 批量处理脚本 - LLVIP(50), MSRS(361), TNO(25)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()) + ['all'],
        default=['all'],
        help='要处理的数据集 (默认: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='输出根目录 (默认: output)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='跳过已经处理过的图像 (默认: True)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='重新处理所有图像（不跳过已存在的）'
    )

    args = parser.parse_args()

    # 处理跳过选项
    skip_existing = not args.no_skip

    print("🚀 FusionINV 批量处理工具")
    print(f"📁 输出目录: {args.output}/")
    print(f"⏭️ 跳过已处理: {'是' if skip_existing else '否'}")
    print(f"🖥️ GPU:  RTX 4090 D")

    # 确定要处理的数据集
    if 'all' in args.datasets:
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = args.datasets

    # 统计总数
    total_pairs = sum(DATASETS[name]['count'] for name in datasets_to_process)
    print(f"📊 预计处理:  {len(datasets_to_process)} 个数据集, 共 {total_pairs} 对图像")

    # 估算时间
    estimated_time = total_pairs * 1  # 假设每对1分钟
    print(f"⏱️ 预计耗时: {estimated_time // 60}小时{estimated_time % 60}分钟 (顺序处理)")
    print()

    # 开始处理
    start_time = time.time()
    overall_results = {'success': 0, 'failed': 0, 'skip': 0, 'timeout': 0, 'error': 0}

    for dataset_name in datasets_to_process:
        if dataset_name in DATASETS:
            results = process_dataset(
                dataset_name,
                DATASETS[dataset_name],
                args.output,
                skip_existing
            )

            # 累加结果
            for key in overall_results:
                overall_results[key] += results[key]

    # 计算总耗时
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # 打印总结
    print(f"\n{'=' * 70}")
    print("🎉 所有数据集处理完成！")
    print(f"{'=' * 70}")
    print(f"📊 总体结果:")
    print(f"  ✅ 成功: {overall_results['success']}")
    print(f"  ⏭️ 跳过:  {overall_results['skip']}")
    print(f"  ❌ 失败: {overall_results['failed']}")
    print(f"  ⏱️ 超时:  {overall_results['timeout']}")
    print(f"  ❗ 错误: {overall_results['error']}")
    print(f"\n⏱️ 总耗时:  {hours}小时{minutes}分钟{seconds}秒")
    print(f"📁 结果保存在: {args.output}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
