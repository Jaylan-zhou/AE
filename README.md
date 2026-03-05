# 人脸重建自编码器实验

DTS410TC 生成式人工智能课程实验 1：使用卷积自编码器实现人脸图像重建。

## 项目概述

本项目实现了一个完整的卷积自编码器（Convolutional Autoencoder），用于 CelebA 人脸数据集的图像重建。通过这个实验，我们能够深入理解潜在表示学习（latent representation learning）和图像重建的原理。

### 核心功能

- ✅ **数据处理**：CelebA 数据集加载、预处理（64×64 归一化）
- ✅ **模型架构**：编码器-解码器结构，支持可配置的潜在维度
- ✅ **训练流程**：完整的训练循环，支持检查点保存
- ✅ **评估工具**：重建质量对比、损失曲线可视化
- ⏳ **对比实验**：不同潜在维度的对比（32、100、256）
- ⏳ **潜在空间分析**：t-SNE 可视化、插值实验

## 项目结构

```
lab1/
├── data/               # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py      # CelebADataset 类
│   └── get_dataloaders.py
├── models/             # 模型定义
│   ├── __init__.py
│   └── autoencoder.py  # Autoencoder, Encoder, Decoder
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── set_seed.py     # 随机种子设置
│   ├── device.py       # 设备检测
│   └── visualization.py # 可视化函数
├── configs/            # 配置
│   ├── __init__.py
│   └── config.py       # 超参数配置
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── requirements.txt    # 依赖列表
└── img_align_celeba/   # CelebA 数据集
```

## 安装说明

### 方法 1：使用 uv（推荐）

**安装 uv**（如果尚未安装）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**创建虚拟环境并安装依赖**：
```bash
# 创建虚拟环境
uv venv

# 安装依赖（自动激活 .venv）
uv pip install -r requirements.txt
```

**环境测试**：
```bash
uv run test_environment.py
```

### 方法 2：使用传统 venv

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**主要依赖：**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0
- tqdm >= 4.65.0

## 使用方法

### 快速测试（推荐先运行）

验证训练流程是否正常：
```bash
# 使用 uv
uv run quick_test.py

# 或使用虚拟环境
python quick_test.py
```

这将使用少量数据训练 3 个 epoch，快速验证所有功能。

### 完整训练

**方法 1：使用脚本（推荐）**：
```bash
# 使用便捷脚本（自动激活环境）
./run.sh

# 或自定义参数
./run.sh --latent_dim 100 --epochs 25
```

**方法 2：直接运行**：
```bash
# 使用 uv
uv run train.py

# 使用虚拟环境
python train.py

# 自定义训练
uv run train.py --latent_dim 100 --epochs 25 --batch_size 64 --lr 0.001
```

**参数说明：**
- `--latent_dim`: 潜在向量维度（默认：100）
- `--epochs`: 训练轮数（默认：25）
- `--batch_size`: 批次大小（默认：64）
- `--lr`: 学习率（默认：0.001）

### 评估模型

```bash
python evaluate.py --checkpoint outputs/models/final_model.pth
```

**指定保存路径：**
```bash
python evaluate.py --checkpoint outputs/models/final_model.pth --save_path outputs/figures/my_reconstruction.png
```

## 模型架构

### 编码器（Encoder）
```
输入 (3, 64, 64)
  ↓ Conv2d(3→32, k=3, s=2) + ReLU
  → (32, 32, 32)
  ↓ Conv2d(32→64, k=3, s=2) + ReLU
  → (64, 16, 16)
  ↓ Conv2d(64→128, k=3, s=2) + ReLU
  → (128, 8, 8)
  ↓ Conv2d(128→256, k=3, s=2) + ReLU
  → (256, 4, 4)
  ↓ Flatten + Linear(4096→latent_dim)
  → 潜在向量 (latent_dim,)
```

### 解码器（Decoder）
```
潜在向量 (latent_dim,)
  ↓ Linear(latent_dim→4096)
  → (256, 4, 4)
  ↓ ConvTranspose2d(256→128, k=4, s=2) + ReLU
  → (128, 8, 8)
  ↓ ConvTranspose2d(128→64, k=4, s=2) + ReLU
  → (64, 16, 16)
  ↓ ConvTranspose2d(64→32, k=4, s=2) + ReLU
  → (32, 32, 32)
  ↓ ConvTranspose2d(32→3, k=4, s=2) + Sigmoid
  → 重建图像 (3, 64, 64)
```

## 训练配置

**超参数：**
- 损失函数：MSE (均方误差)
- 优化器：AdamW
- 学习率：0.001
- 权重衰减：0.0001
- Batch Size：64
- Epochs：25

**数据划分：**
- 训练集：80%
- 验证集：20%

## 输出文件

训练后会生成以下文件：

```
outputs/
├── models/
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   ├── ...
│   ├── checkpoint_epoch_25_best.pth  # 最佳模型
│   └── final_model.pth               # 最终模型
└── figures/
    └── loss_curve.png                # 损失曲线
```

## 实验任务

根据实验要求，本项目需要完成以下任务：

### ✅ 已完成

1. **数据准备与可视化**
   - [x] 加载 CelebA 数据集
   - [x] 图像预处理（64×64，归一化到 [0,1]）
   - [x] 创建训练/验证 DataLoader
   - [x] 批次图像可视化

2. **模型架构设计**
   - [x] 编码器：4 层卷积网络
   - [x] 解码器：4 层转置卷积
   - [x] 可配置潜在维度

3. **训练流程**
   - [x] MSE 损失函数
   - [x] AdamW 优化器
   - [x] 25 epoch 训练
   - [x] 损失曲线可视化

4. **评估工具**
   - [x] 重建质量对比
   - [x] MSE 指标计算

### ⏳ 待完成

5. **对比实验**
   - [ ] 训练不同潜在维度模型（32、100、256）
   - [ ] 对比重建质量
   - [ ] 分析维度影响

6. **潜在空间分析**
   - [ ] t-SNE 降维可视化
   - [ ] 潜在向量插值
   - [ ] 统计分析

7. **随机生成分析**
   - [ ] 从随机向量生成图像
   - [ ] 分析生成局限性

8. **实验报告**
   - [ ] 回答三个核心问题
   - [ ] 生成完整报告

## 核心问题

### 问题 1：潜在空间存储了什么信息？

潜在空间编码了人脸的关键特征，包括：
- 身份特征（眼睛、鼻子、嘴巴的形状和位置）
- 表情特征（微笑、皱眉等）
- 视角特征（正面、侧面等）
- 光照条件

### 问题 2：为什么自编码器能重建但不能可靠生成？

- **重建能力强**：训练时只覆盖数据流形上的点
- **生成能力弱**：
  - 潜在空间不连续，存在"空洞"
  - 随机采样可能落在空白区域
  - 缺乏正则化（如 VAE 的 KL 散度）
  - 无法保证先验分布匹配

### 问题 3：潜在维度如何影响重建质量？

- **较小维度（32）**：
  - 优点：压缩率高，计算快
  - 缺点：可能丢失细节，重建模糊

- **中等维度（100）**：
  - 平衡点，质量与效率兼顾

- **较大维度（256）**：
  - 优点：重建质量高，细节丰富
  - 缺点：参数多，易过拟合，计算慢

## 性能指标

在标准配置下（latent_dim=100, epochs=25）：

- **模型参数量**：约 1.4M
- **训练时间**：约 30-60 分钟（取决于 GPU）
- **最终 MSE**：约 0.01-0.03（取决于数据集大小）

## 故障排除

### CUDA 内存不足

**解决方案**：
1. 减小 batch_size：`--batch_size 32`
2. 使用 CPU 训练（较慢）

### 中文字体显示问题

如果图表中中文显示为方块：

**Mac OS**：
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```

**Windows**：
```python
plt.rcParams['font.sans-serif'] = ['SimHei']
```

**Linux**：
```bash
sudo apt-get install fonts-wqy-microhei
```

## 扩展方向

1. **变分自编码器（VAE）**：改进生成能力
2. **条件自编码器**：基于属性的图像生成
3. **超分辨率**：放大低分辨率图像
4. **图像去噪**：自动去除图像噪声

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [CelebA 数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Autoencoder - Wikipedia](https://en.wikipedia.org/wiki/Autoencoder)

## 许可证

本项目仅用于教学目的。

## 作者

DTS410TC 生成式人工智能课程实验
