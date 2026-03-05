"""
设备检测和配置
"""
import torch


def get_device():
    """
    自动检测并返回可用的计算设备（CPU 或 GPU）

    Returns:
        torch.device: 可用的计算设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device
