# 人脸重建自编码器技术设计

## Context

### 背景与现状

本项目是 DTS410TC 生成式人工智能课程的实验 1，旨在实现一个卷积自编码器用于人脸图像重建。项目从零开始，不依赖现有代码库，需要构建完整的数据处理、模型训练和评估流程。

### 关键约束

1. **数据集位置**：数据已存在于 `./img_align_celeba/` 目录，无需下载
2. **计算资源**：需支持 CPU 和 GPU 训练，自动检测可用设备
3. **时间限制**：训练 20-30 epoch，需在合理时间内完成
4. **实验要求**：必须生成完整的可视化和分析报告，回答三个核心问题

### 利益相关者

- 课程助教：评估实验完成度和代码质量
- 学生本人：理解潜在表示学习原理，为后续 VAE/GAN 学习打基础

## Goals / Non-Goals

### Goals

1. ✅ 实现符合规格的卷积自编码器架构（编码器 + 解码器）
2. ✅ 建立端到端的训练流程（数据加载 → 训练 → 验证 → 保存）
3. ✅ 生成高质量的可视化结果（重建对比、损失曲线、潜在空间）
4. ✅ 完成不同潜在维度的对比实验（32、100、256）
5. ✅ 提供深入的实验分析，回答三个核心问题

### Non-Goals

- ❌ 实现数据增强（随机翻转、旋转等）- 仅基础预处理
- ❌ 支持分布式训练或多 GPU 并行
- ❌ 实现高级生成模型（VAE、GAN、Diffusion）
- ❌ 构建生产级部署方案（Web 服务、API 接口）
- ❌ 优化推理速度（模型量化、剪枝、TensorRT 等）

## Decisions

### 1. 项目结构设计

**决策**：采用模块化 Python 项目结构

```
lab1/
├── data/               # 数据处理模块
│   ├── __init__.py
│   └── dataset.py      # Dataset 类和 DataLoader 创建
├── models/             # 模型定义模块
│   ├── __init__.py
│   └── autoencoder.py  # Autoencoder、Encoder、Decoder 类
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── visualization.py # 可视化函数
│   └── metrics.py      # 评估指标
├── configs/            # 配置文件
│   └── config.py       # 超参数配置
├── outputs/            # 输出目录
│   ├── models/         # 模型权重
│   ├── figures/        # 图表
│   └── reports/        # 实验报告
└── img_align_celeba/   # 数据集（已存在）
```

**理由**：
- 模块化设计便于代码维护和测试
- 清晰的职责分离（数据、模型、训练、评估）
- 符合 PyTorch 项目最佳实践
- 便于后续扩展（如添加新模型或实验）

**备选方案**：
- 单文件脚本（~500 行）→ ❌ 难以维护和测试
- Jupyter Notebook → ✅ 可作为补充，但不适合大型项目

---

### 2. 模型架构实现

**决策**：使用 PyTorch `nn.Module` 分别实现 Encoder、Decoder 和 Autoencoder

**Encoder 设计**：
```python
class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        # Conv1: 3 → 32, kernel=3, stride=2, padding=1
        # Conv2: 32 → 64, kernel=3, stride=2, padding=1
        # Conv3: 64 → 128, kernel=3, stride=2, padding=1
        # Conv4: 128 → 256, kernel=3, stride=2, padding=1
        # FC: 256*4*4 → latent_dim
```

**Decoder 设计**：
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=100):
        # FC: latent_dim → 256*4*4
        # ConvTranspose1: 256 → 128, kernel=4, stride=2, padding=1
        # ConvTranspose2: 128 → 64, kernel=4, stride=2, padding=1
        # ConvTranspose3: 64 → 32, kernel=4, stride=2, padding=1
        # ConvTranspose4: 32 → 3, kernel=4, stride=2, padding=1
```

**理由**：
- 转置卷积（ConvTranspose2d）比双线性插值 + 卷积更高效
- kernel_size=4 配合 stride=2、padding=1 可精确倍增分辨率
- 分离 Encoder/Decoder 便于独立使用（如仅提取特征）

**备选方案**：
- 使用 `nn.Upsample` + 卷积 → 更稳定但计算量更大
- 使用 ResNet 架构 → 过于复杂，不符合实验要求

---

### 3. 损失函数与优化器

**决策**：
- 损失函数：`nn.MSELoss(reduction='mean')`
- 优化器：`AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)`

**理由**：
- MSE 是重建任务的标准损失函数
- AdamW 结合了 Adam 的自适应学习率和权重衰减（L2 正则化）
- 学习率 0.001 是 Adam 的经典默认值，适合此规模模型

**备选方案**：
- SGD + Momentum → 需要手动调参，收敛慢
- L1 Loss → 对异常值更鲁棒，但训练不稳定
- Perceptual Loss (VGG) → 过于复杂，不符合基础实验要求

---

### 4. 训练策略

**决策**：
- Epoch 数量：25（平衡训练时间和效果）
- Batch Size：64（标准值，适合 64×64 图像）
- 验证频率：每个 epoch 结束后验证
- 模型保存：
  - 最佳模型：验证损失最低时保存
  - 最终模型：训练结束后保存
  - 检查点：每 5 epoch 保存一次（可选）

**理由**：
- 25 epoch 足以让模型收敛（观察到损失曲线平稳）
- Batch size 64 在 GPU 内存和梯度稳定性之间取得平衡
- 每个验证模型便于早期发现过拟合

**备选方案**：
- 学习率调度器（ReduceLROnPlateau）→ 可选优化
- 早停法（Early Stopping）→ 有价值，但实验要求固定 epoch

---

### 5. 数据预处理流程

**决策**：使用 `torchvision.transforms.Compose`

```python
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # 自动归一化到 [0, 1]
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
```

**理由**：
- `ToTensor()` 自动将 [0, 255] 转换为 [0, 1] 的 float32
- 不使用额外的归一化（如 mean=[0.5, 0.5, 0.5]），保持简单
- 训练集和验证集使用相同的变换（无数据增强）

**数据集划分**：
- 训练集：80%（使用 `random_split` 或直接索引）
- 验证集：20%
- 固定随机种子确保可复现

---

### 6. 可视化方案

**决策**：使用 matplotlib 绘制所有图表

**可视化类型**：
1. **批次数据可视化**：8×8 网格显示 64 张训练图像
2. **损失曲线**：训练和验证损失随 epoch 变化（双线图）
3. **重建对比**：左右或上下对比原始/重建图像
4. **潜在空间可视化**：
   - t-SNE 降维到 2D 散点图
   - 潜在向量插值序列（10 张图像）
5. **不同维度对比**：并排显示 32/100/256 维的重建结果

**可视化配置**：
```python
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
plt.figure(figsize=(12, 8))                   # 高分辨率
```

**理由**：
- matplotlib 是 PyTorch 生态的标准可视化库
- 支持中文标签（需配置字体）
- 可保存为高分辨率 PNG/PDF

---

### 7. 实验可复现性

**决策**：在所有脚本中设置随机种子

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**配置记录**：
- 保存完整超参数到 JSON 文件
- 记录环境信息（Python、PyTorch、CUDA 版本）
- 每次实验生成时间戳文件夹

**理由**：
- 确保实验可重复（课程要求）
- 便于调试和对比不同运行结果
- 符合科研最佳实践

---

### 8. 潜在空间分析方法

**决策**：实现多层次分析

1. **统计分析**：
   - 计算每个潜在维度的均值、标准差
   - 分析潜在向量的 L2 范数分布

2. **降维可视化**：
   - 使用 `sklearn.manifold.TSNE` 降维到 2D
   - 绘制散点图，按属性着色（如有标签）

3. **插值实验**：
   - 在两个样本间线性插值
   - 解码插值点生成图像序列
   - 观察潜在空间的连续性

4. **随机生成测试**：
   - 从 N(0,1) 采样随机向量
   - 解码生成图像
   - 评估质量并分析失败原因

**理由**：
- 多角度分析有助于回答实验问题
- t-SNE 可视化直观展示潜在空间结构
- 插值实验揭示空间的连续性和语义性

---

## Risks / Trade-offs

### Risk 1: GPU 内存不足

**风险描述**：CelebA 数据集较大（~200K 图像），batch_size=64 可能超出 GPU 内存

**缓解措施**：
- 自动检测 GPU 内存，动态调整 batch_size
- 提供 CPU 备选方案
- 使用梯度累积模拟大 batch

### Risk 2: 训练时间过长

**风险描述**：25 epoch 可能在单 GPU 上耗时数小时

**缓解措施**：
- 提供进度条（`tqdm`）和 ETA 估计
- 支持从检查点恢复训练
- 提供快速测试模式（少量数据，少量 epoch）

### Risk 3: 过拟合

**风险描述**：模型可能过度记忆训练数据，验证损失上升

**缓解措施**：
- 监控验证损失，实施早停（可选）
- 增加权重衰减（weight_decay=0.0001）
- 在实验报告中分析过拟合现象

### Risk 4: 可视化中文显示问题

**风险描述**：matplotlib 默认不支持中文，显示为方块

**缓解措施**：
- 提供字体检测和回退机制
- 如果系统中文字体不可用，使用英文标签
- 提供字体安装指南

### Risk 5: 数据集路径问题

**风险描述**：用户可能将数据集放在不同位置

**缓解措施**：
- 在配置文件中提供 `data_dir` 参数
- 启动时自动检测数据集位置
- 提供明确的错误提示

### Trade-off 1: 模型复杂度 vs 训练时间

**权衡**：更深的网络（5-6 层）可能提高重建质量，但训练更慢

**选择**：遵循实验要求的 4 层架构，平衡效果和效率

### Trade-off 2: 潜在维度大小

**权衡**：
- 更大维度（256）→ 更好重建，但更多存储和计算
- 更小维度（32）→ 更高效，但可能丢失细节

**选择**：通过对比实验（32/100/256）量化权衡

---

## Migration Plan

### 阶段 1：环境准备（预计 10 分钟）

1. 创建虚拟环境并安装依赖
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install torch torchvision matplotlib seaborn numpy scikit-learn tqdm
   ```

2. 验证数据集路径
   ```bash
   ls ./img_align_celeba/ | head  # 确认图像存在
   ```

### 阶段 2：基础实现（预计 2-3 小时）

1. 实现 `data/dataset.py`：Dataset 类和 DataLoader
2. 实现 `models/autoencoder.py`：Encoder、Decoder、Autoencoder
3. 实现 `train.py`：基础训练循环
4. 测试单个 epoch 训练

### 阶段 3：完整训练流程（预计 1-2 小时）

1. 实现模型检查点保存和加载
2. 添加损失曲线可视化
3. 完整训练 25 epoch
4. 验证重建质量

### 阶段 4：评估与分析（预计 2-3 小时）

1. 实现 `evaluate.py`：重建对比可视化
2. 实现不同潜在维度对比实验
3. 实现潜在空间分析（t-SNE、插值）
4. 生成实验报告（Markdown）

### 阶段 5：验证与提交（预计 30 分钟）

1. 确保所有可视化图表生成
2. 检查实验报告完整性
3. 运行最终验证测试
4. 准备提交材料

### 回滚策略

- 每个阶段完成后提交代码到 Git（本地或远程）
- 保留每个阶段的输出文件夹（时间戳命名）
- 如果训练失败，可从最近检查点恢复

---

## Open Questions

### Q1: 是否需要实现数据增强？

**状态**：❌ 不需要

**理由**：实验要求中未提及数据增强，且基础预处理已足够。如有时间可添加可选的随机翻转作为扩展实验。

### Q2: 是否需要支持多 GPU 训练？

**状态**：❌ 不需要

**理由**：实验规模较小，单 GPU 足够。添加 `DataParallel` 会增加代码复杂度，不符合课程要求。

### Q3: 如何处理数据集不平衡问题？

**状态**：✅ 无需处理

**理由**：CelebA 是人脸数据集，天然平衡（每个人一张图），无需类别平衡策略。

### Q4: 是否需要实现评估指标（PSNR、SSIM）？

**状态**：🤚 可选

**理由**：MSE 损失已足够评估重建质量。PSNR/SSIM 可作为补充，增强实验报告的完整性。

### Q5: 如何解释自编码器生成能力有限的原因？

**状态**：✅ 需要在实验报告中回答

**核心观点**：
- 自编码器的潜在空间不连续且稀疏
- 训练时只覆盖数据流形，随机采样可能落在空白区域
- 缺乏正则化（如 VAE 的 KL 散度），无法保证先验分布匹配
- **解决方案**：VAE、GAN 等生成模型

---

## Performance Considerations

### 训练速度优化

1. **数据加载优化**：
   - 使用 `num_workers=4` 并行加载
   - 启用 `pin_memory=True` 加速 GPU 传输

2. **混合精度训练**（可选）：
   ```python
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       loss = criterion(output, target)
   ```

3. **梯度累积**（模拟大 batch）：
   ```python
   accumulation_steps = 4
   if (i + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

### 内存优化

1. **清理不需要的中间变量**：
   ```python
   del intermediate_tensor
   torch.cuda.empty_cache()
   ```

2. **使用更小的 batch size**（如果 OOM）：
   ```python
   batch_size = 32  # 从 64 降到 32
   ```

---

## Testing Strategy

### 单元测试（可选）

1. **数据加载测试**：验证 DataLoader 返回正确的形状
2. **模型前向传播测试**：验证输入输出形状匹配
3. **损失函数测试**：验证 MSE 计算正确性

### 集成测试

1. **端到端训练测试**：在少量数据上训练 1 个 epoch
2. **检查点测试**：保存并加载模型，验证输出一致
3. **可视化测试**：确认所有图表正常生成

### 验收标准

- [ ] 训练损失收敛（损失曲线平稳）
- [ ] 验证损失与训练损失接近（无明显过拟合）
- [ ] 重建图像清晰可辨（关键特征保留）
- [ ] 所有可视化图表生成且清晰
- [ ] 实验报告完整回答三个问题

---

## Documentation Plan

### 代码文档

- 所有公共函数添加 docstring（Google 风格）
- 关键算法添加注释说明
- 复杂操作添加示例代码

### 用户文档

- `README.md`：项目介绍、安装说明、使用指南
- `EXPERIMENT_REPORT.md`：实验报告（自动生成 + 手工补充）
- `CONFIG.md`：配置参数说明（可选）

### 示例用法

```bash
# 训练模型
python train.py --latent_dim 100 --epochs 25 --batch_size 64

# 评估模型
python evaluate.py --model_path outputs/models/best_model.pth

# 对比不同潜在维度
python evaluate.py --compare_latent_dims 32 100 256
```
