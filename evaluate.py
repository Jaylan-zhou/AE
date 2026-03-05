"""
评估脚本
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config
from utils import get_device
from utils.visualization import visualize_reconstruction
from data import CelebADataset, get_transforms, get_dataloaders
from models import Autoencoder


def load_model(checkpoint_path, device):
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 设备

    Returns:
        model: 加载好的模型
        checkpoint: 检查点数据
    """
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取潜在维度
    if 'model_state_dict' in checkpoint:
        # 尝试从模型状态推断
        state_dict = checkpoint['model_state_dict']
        # 从编码器全连接层推断潜在维度 (weight shape: (latent_dim, 4096))
        latent_dim = state_dict['encoder.fc.weight'].shape[0]
    else:
        latent_dim = 100  # 默认值

    # 创建模型
    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功 (潜在维度: {latent_dim})")

    return model, checkpoint


def compute_mse(model, dataloader, device):
    """
    计算 MSE 损失

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        float: 平均 MSE 损失
    """
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_reconstruction(model, val_loader, device, save_path=None, num_images=8):
    """
    评估重建质量并生成对比图

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        save_path: 保存路径
        num_images: 显示的图像数量
    """
    # 获取一个批次
    images = next(iter(val_loader))
    images = images[:num_images].to(device)

    # 生成重建图像
    with torch.no_grad():
        reconstructed = model(images)

    # 转回 CPU 用于可视化
    images_cpu = images.cpu()
    reconstructed_cpu = reconstructed.cpu()

    # 可视化
    if save_path:
        visualize_reconstruction(images_cpu, reconstructed_cpu, save_path, num_images)
    else:
        visualize_reconstruction(images_cpu, reconstructed_cpu, num_images=num_images)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='评估自编码器')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存重建对比图的路径')

    args = parser.parse_args()

    # 设置设备
    device = get_device()

    # 加载模型
    model, checkpoint = load_model(args.checkpoint, device)

    # 创建数据加载器
    print("\n准备数据...")
    _, val_loader = get_dataloaders()

    # 计算 MSE
    print("\n计算评估指标...")
    mse = compute_mse(model, val_loader, device)
    print(f"MSE: {mse:.6f}")

    # 生成重建对比图
    print("\n生成重建对比图...")
    if args.save_path is None:
        args.save_path = "./outputs/figures/reconstruction_comparison.png"
    evaluate_reconstruction(model, val_loader, device, args.save_path)

    print("\n✓ 评估完成！")


if __name__ == "__main__":
    main()
