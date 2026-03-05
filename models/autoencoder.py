"""
卷积自编码器模型
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    编码器：将图像压缩为潜在向量
    """

    def __init__(self, latent_dim=100):
        """
        Args:
            latent_dim (int): 潜在向量维度
        """
        super(Encoder, self).__init__()

        # 4 层卷积网络
        # 输入: (batch, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # -> (batch, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (batch, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> (batch, 128, 8, 8)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> (batch, 256, 4, 4)

        self.relu = nn.ReLU(inplace=True)

        # 全连接层: 256*4*4 -> latent_dim
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """使用 Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x (tensor): 输入图像 (batch, 3, 64, 64)

        Returns:
            tensor: 潜在向量 (batch, latent_dim)
        """
        # 卷积层
        x = self.relu(self.conv1(x))  # (batch, 32, 32, 32)
        x = self.relu(self.conv2(x))  # (batch, 64, 16, 16)
        x = self.relu(self.conv3(x))  # (batch, 128, 8, 8)
        x = self.relu(self.conv4(x))  # (batch, 256, 4, 4)

        # 展平
        x = x.view(x.size(0), -1)  # (batch, 256*4*4)

        # 全连接层
        x = self.fc(x)  # (batch, latent_dim)

        return x


class Decoder(nn.Module):
    """
    解码器：从潜在向量重建图像
    """

    def __init__(self, latent_dim=100):
        """
        Args:
            latent_dim (int): 潜在向量维度
        """
        super(Decoder, self).__init__()

        # 全连接层: latent_dim -> 256*4*4
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        # 4 层转置卷积
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> (batch, 128, 8, 8)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> (batch, 64, 16, 16)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> (batch, 32, 32, 32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # -> (batch, 3, 64, 64)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """使用 Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x (tensor): 潜在向量 (batch, latent_dim)

        Returns:
            tensor: 重建图像 (batch, 3, 64, 64)
        """
        # 全连接层
        x = self.fc(x)  # (batch, 256*4*4)

        # 重塑为特征图
        x = x.view(x.size(0), 256, 4, 4)  # (batch, 256, 4, 4)

        # 转置卷积层
        x = self.relu(self.deconv1(x))  # (batch, 128, 8, 8)
        x = self.relu(self.deconv2(x))  # (batch, 64, 16, 16)
        x = self.relu(self.deconv3(x))  # (batch, 32, 32, 32)
        x = self.sigmoid(self.deconv4(x))  # (batch, 3, 64, 64)

        return x


class Autoencoder(nn.Module):
    """
    自编码器：组合编码器和解码器
    """

    def __init__(self, latent_dim=100):
        """
        Args:
            latent_dim (int): 潜在向量维度
        """
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x (tensor): 输入图像 (batch, 3, 64, 64)

        Returns:
            tensor: 重建图像 (batch, 3, 64, 64)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_encoder(self):
        """获取编码器"""
        return self.encoder

    def get_decoder(self):
        """获取解码器"""
        return self.decoder


def count_parameters(model):
    """
    计算模型参数量

    Args:
        model: 模型

    Returns:
        int: 参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
