"""
配置文件：定义所有超参数和路径配置
"""


class Config:
    """超参数配置类"""

    # 数据路径
    data_dir = "./img_align_celeba"
    train_split = 0.8  # 训练集比例

    # 模型超参数
    latent_dim = 100  # 潜在向量维度
    image_size = 64  # 图像大小

    # 训练超参数
    batch_size = 64
    num_epochs = 25
    learning_rate = 0.001
    weight_decay = 0.0001

    # 优化器配置
    optimizer = "AdamW"

    # 输出路径
    output_dir = "./outputs"
    models_dir = f"{output_dir}/models"
    figures_dir = f"{output_dir}/figures"
    reports_dir = f"{output_dir}/reports"

    # 设备配置
    device = "cuda"  # 会在运行时自动检测

    # 随机种子
    seed = 42

    # 可视化配置
    num_images_to_visualize = 64  # 8x8 网格
    figure_dpi = 100

    # 中文显示配置
    use_chinese = True
    chinese_font = "SimHei"  # Windows: SimHei, Mac: Arial Unicode MS
