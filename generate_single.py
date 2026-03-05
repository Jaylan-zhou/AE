#!/usr/bin/env python3
"""
单张图片生成脚本：使用训练好的自编码器对单张图片进行重建，并生成原图与重建图的对比图。

用法示例：
    # 使用数据集中的图片
    python generate_single.py --model_path outputs/models/final_model.pth --image_path img_align_celeba/000001.jpg

    # 使用任意图片路径（需先将 Windows 上训练的模型复制到本机）
    python generate_single.py --model_path path/to/final_model.pth --image_path /path/to/your/image.jpg

    # 指定输出路径
    python generate_single.py --model_path outputs/models/final_model.pth --image_path img_align_celeba/000001.jpg --output outputs/figures/my_comparison.png
"""
import os
import argparse
import torch
from PIL import Image

from configs.config import Config
from utils import get_device
from utils.visualization import setup_chinese_font
from data.dataset import get_transforms
from models import Autoencoder
import matplotlib.pyplot as plt
import numpy as np


def load_model(checkpoint_path, device):
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 设备

    Returns:
        model: 加载好的模型
    """
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取潜在维度（从编码器全连接层推断）
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # encoder.fc: Linear(4096, latent_dim), weight shape = (latent_dim, 4096)
        latent_dim = state_dict['encoder.fc.weight'].shape[0]
    else:
        latent_dim = 100  # 默认值

    # 创建模型并加载权重
    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功 (潜在维度: {latent_dim})")
    return model


def load_and_preprocess_image(image_path, transform):
    """
    加载并预处理单张图片

    Args:
        image_path: 图片路径
        transform: 图像变换（需与训练时一致）

    Returns:
        tensor: 预处理后的图像 (1, 3, 64, 64)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    # 添加 batch 维度
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def visualize_single_comparison(original, reconstructed, save_path=None, show=True):
    """
    可视化单张图片的原图与重建图对比（左右并排）

    Args:
        original: 原图 tensor (1, 3, 64, 64)
        reconstructed: 重建图 tensor (1, 3, 64, 64)
        save_path: 保存路径（可选）
        show: 是否弹出显示窗口
    """
    setup_chinese_font()

    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()

    # 取第一张图，(C, H, W) -> (H, W, C)
    orig_img = original[0].transpose(1, 2, 0)
    recon_img = reconstructed[0].transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('原图 vs 重建图 对比', fontsize=14)

    axes[0].imshow(orig_img)
    axes[0].set_title('原图', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(recon_img)
    axes[1].set_title('重建图', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存到: {save_path}")

    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='使用训练好的自编码器对单张图片进行重建，并生成原图与重建图对比'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='outputs/models/final_model.pth',
        help='模型检查点路径（默认: outputs/models/final_model.pth）'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='输入图片路径（支持 jpg、jpeg、png）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/figures/single_reconstruction.png',
        help='输出对比图保存路径（默认: outputs/figures/single_reconstruction.png）'
    )
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='不弹出显示窗口，仅保存文件'
    )

    args = parser.parse_args()

    # 设置设备
    device = get_device()

    # 加载模型
    model = load_model(args.model_path, device)

    # 准备图像变换（与训练时一致）
    transform = get_transforms(Config.image_size)

    # 加载并预处理图片
    print(f"\n加载图片: {args.image_path}")
    image_tensor = load_and_preprocess_image(args.image_path, transform)
    image_tensor = image_tensor.to(device)

    # 生成重建图
    print("生成重建图...")
    with torch.no_grad():
        reconstructed = model(image_tensor)

    # 转回 CPU 用于可视化
    original_cpu = image_tensor.cpu()
    reconstructed_cpu = reconstructed.cpu()

    # 可视化对比
    visualize_single_comparison(
        original_cpu, reconstructed_cpu,
        save_path=args.output,
        show=not args.no_show
    )

    print("\n✓ 完成！")


if __name__ == '__main__':
    main()
