"""
CelebA 数据集类
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    """
    CelebA 人脸数据集类
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 数据集目录路径
            transform (callable, optional): 图像转换操作
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取所有图像文件路径
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"在 {root_dir} 中未找到图像文件")

        print(f"加载了 {len(self.image_files)} 张图像")

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx (int): 样本索引

        Returns:
            tensor: 转换后的图像张量
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_transforms(image_size=64):
    """
    获取数据转换操作

    Args:
        image_size (int): 目标图像大小

    Returns:
        transforms.Compose: 图像转换操作
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # 自动将 [0, 255] 转换为 [0, 1]
    ])
