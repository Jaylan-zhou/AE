"""
训练脚本
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.config import Config
from utils import set_seed, get_device
from utils.visualization import plot_loss_curve
from data import get_dataloaders
from models import Autoencoder, count_parameters


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备

    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="训练", leave=False)

    for images in progress_bar:
        images = images.to(device)

        # 前向传播
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    验证模型

    Args:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        float: 平均损失
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证", leave=False)

        for images in progress_bar:
            images = images.to(device)

            # 前向传播
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            total_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_path, is_best=False):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 损失值
        save_path: 保存路径
        is_best: 是否为最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 最佳模型已保存到: {best_path}")


def train(config=None):
    """
    训练主函数

    Args:
        config: 配置对象
    """
    if config is None:
        config = Config()

    # 设置随机种子和设备
    set_seed(config.seed)
    device = get_device()

    # 创建输出目录
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # 创建数据加载器
    print("\n" + "="*50)
    print("准备数据...")
    print("="*50)
    train_loader, val_loader = get_dataloaders(config)

    # 创建模型
    print("\n" + "="*50)
    print("创建模型...")
    print("="*50)
    model = Autoencoder(latent_dim=config.latent_dim).to(device)
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 训练循环
    print("\n" + "="*50)
    print(f"开始训练 ({config.num_epochs} epochs)")
    print("="*50)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    total_start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

        print(f"\nEpoch [{epoch}/{config.num_epochs}]")
        print("-" * 50)

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 计算时间
        epoch_time = time.time() - epoch_start_time

        # 打印结果
        print(f"训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 耗时: {epoch_time:.2f}秒")

        # 保存检查点
        checkpoint_path = f"{config.models_dir}/checkpoint_epoch_{epoch}.pth"
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        if epoch % 5 == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, is_best)

    # 保存最终模型
    final_path = f"{config.models_dir}/final_model.pth"
    save_checkpoint(model, optimizer, config.num_epochs, val_losses[-1], final_path)

    total_time = time.time() - total_start_time
    print("\n" + "="*50)
    print(f"训练完成！总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("="*50)

    # 绘制损失曲线
    print("\n生成损失曲线...")
    loss_curve_path = f"{config.figures_dir}/loss_curve.png"
    plot_loss_curve(train_losses, val_losses, loss_curve_path)

    print("\n✓ 所有训练完成！")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='训练自编码器')

    parser.add_argument('--latent_dim', type=int, default=100,
                        help='潜在向量维度')
    parser.add_argument('--epochs', type=int, default=25,
                        help='训练 epoch 数量')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 创建配置并更新
    config = Config()
    config.latent_dim = args.latent_dim
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.seed = args.seed

    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
