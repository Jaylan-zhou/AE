"""
可视化工具函数
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def setup_chinese_font():
    """设置中文字体显示"""
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 尝试设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'STHeiti']
    except:
        print("警告：无法设置中文字体，将使用默认字体")


def visualize_batch(images, title="批次图像", save_path=None, nrows=8, ncols=8):
    """
    可视化一个批次的图像

    Args:
        images (tensor): 图像批次张量 (B, C, H, W)
        title (str): 图像标题
        save_path (str): 保存路径（可选）
        nrows (int): 行数
        ncols (int): 列数
    """
    setup_chinese_font()

    # 转换张量为 numpy 数组
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    # 限制图像数量
    num_images = min(len(images), nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    fig.suptitle(title, fontsize=16)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = axes[i, j]

            if idx < num_images:
                # 转换 (C, H, W) -> (H, W, C)
                img = images[idx].transpose(1, 2, 0)
                ax.imshow(img)
            else:
                ax.axis('off')

            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()
    plt.close()


def plot_loss_curve(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        save_path (str): 保存路径（可选）
    """
    setup_chinese_font()

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('训练和验证损失曲线', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"损失曲线已保存到: {save_path}")

    plt.show()
    plt.close()


def visualize_reconstruction(original, reconstructed, save_path=None, num_images=8):
    """
    可视化原始图像和重建图像对比

    Args:
        original (tensor): 原始图像批次
        reconstructed (tensor): 重建图像批次
        save_path (str): 保存路径（可选）
        num_images (int): 显示的图像对数量
    """
    setup_chinese_font()

    # 转换为 numpy
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()

    num_images = min(num_images, len(original))

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    fig.suptitle('重建质量对比', fontsize=14)

    for i in range(num_images):
        # 原始图像
        orig_img = original[i].transpose(1, 2, 0)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('原始', fontsize=10)

        # 重建图像
        recon_img = reconstructed[i].transpose(1, 2, 0)
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('重建', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"重建对比图已保存到: {save_path}")

    plt.show()
    plt.close()


def visualize_latent_comparison(images_dict, save_path=None):
    """
    可视化不同潜在维度的重建结果对比

    Args:
        images_dict (dict): 不同潜在维度的图像字典
            {32: tensor, 100: tensor, 256: tensor}
        save_path (str): 保存路径（可选）
    """
    setup_chinese_font()

    num_dims = len(images_dict)
    num_images = 8

    fig, axes = plt.subplots(num_dims + 1, num_images,
                            figsize=(num_images * 1.5, (num_dims + 1) * 1.5))

    # 第一行：原始图像
    original_key = list(images_dict.keys())[0]
    original = images_dict[original_key]['original']

    for i in range(num_images):
        orig_img = original[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('原始', fontsize=10)

    # 其他行：不同潜在维度的重建
    for row, (dim, data) in enumerate(images_dict.items(), 1):
        reconstructed = data['reconstructed']

        for i in range(num_images):
            recon_img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
            axes[row, i].imshow(recon_img)
            axes[row, i].axis('off')
            if i == 0:
                axes[row, i].set_title(f'维度={dim}', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"潜在维度对比图已保存到: {save_path}")

    plt.show()
    plt.close()
