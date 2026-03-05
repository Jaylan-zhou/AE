"""
随机种子设置函数
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    设置所有随机种子以确保实验可复现

    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保 CuDNN 使用确定性算法（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")
