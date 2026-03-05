"""
数据加载器创建函数
"""
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import CelebADataset, get_transforms
from configs.config import Config


def get_dataloaders(config=None):
    """
    创建训练集和验证集的 DataLoader

    Args:
        config: 配置对象

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    if config is None:
        config = Config()

    # 创建数据集
    transform = get_transforms(config.image_size)
    full_dataset = CelebADataset(config.data_dir, transform=transform)

    # 计算训练集和验证集大小
    total_size = len(full_dataset)
    train_size = int(total_size * config.train_split)
    val_size = total_size - train_size

    # 随机划分数据集
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    return train_loader, val_loader
