import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# 加载KL分数数据 (这里假设你的路径和格式没变)
data_path = Path(__file__).parent / 'kl_scores_export' / 'kl_scores_10imgs.pkl'
try:
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} batches")
except FileNotFoundError:
    print(f"File not found at {data_path}. Please check the path.")
    exit()

# 筛选stage2的数据
stage2_data = [item for item in data_list if item['shape'][1] == 84]
print(f"Stage2 batches: {len(stage2_data)}")

if not stage2_data:
    print("No stage2 data found!")
    exit()

# 合并stage2数据
all_data = np.concatenate([item['data'] for item in stage2_data], axis=0)
print(f"Total stage2 data shape: {all_data.shape}")

num_images = 10
rows_per_image = all_data.shape[0] // num_images
print(f"Rows per image (Tokens per image): {rows_per_image}")

# 为每张图分别画累积质量曲线
for img_idx in range(num_images):
    img_data = all_data[img_idx * rows_per_image : (img_idx + 1) * rows_per_image]
    kl_flat = img_data.flatten()
    
    # ================= 核心数学转换 =================
    # 1. 归一化 KL 分数，将其视为“信息质量 (Information Mass)”
    # 加上 1e-8 防止除以 0

    # ================= 修改后的代码 =================
    kl_min, kl_max = kl_flat.min(), kl_flat.max()
    normalized_kl = (kl_flat - kl_min) / (kl_max - kl_min + 1e-8)

    # 引入指数锐化 (Power Penalty) 消除背景底噪累积
    # γ (gamma) 可以取 2, 3 或 4。这里以 3 为例。
    info_mass = normalized_kl ** 2
    
    # 2. 将质量从大到小排序 (优先保留高价值目标)
    mass_sorted = np.sort(info_mass)[::-1]
    
    # 3. 计算累积信息质量
    cumulative_mass = np.cumsum(mass_sorted)
    total_mass = cumulative_mass[-1]
    
    # 4. 转换为百分比
    cumulative_mass_pct = (cumulative_mass / total_mass) * 100
    token_pct = (np.arange(1, len(mass_sorted) + 1) / len(mass_sorted)) * 100
    
    # 5. 寻找 95% 信息质量达标点
    # 找到第一个累积质量大于等于 95% 的索引
    idx_95 = np.argmax(cumulative_mass_pct >= 95.0)
    tokens_needed_95 = token_pct[idx_95]
    
    print(f"\nImage {img_idx}:")
    print(f"  Total Tokens: {len(kl_flat)}")
    print(f"  To preserve 95% of KL Mass, we only need {tokens_needed_95:.2f}% of Tokens!")
    
    # ================= 绘图 =================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：归一化后的 KL 质量分布直方图 (看长尾效应)
    axes[0].hist(info_mass, bins=50, alpha=0.7, color='royalblue', edgecolor='black')
    axes[0].set_title(f'Image {img_idx} - Info Mass Distribution (Normalized KL)', fontsize=12)
    axes[0].set_xlabel('Information Mass [0, 1]')
    axes[0].set_ylabel('Token Count')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # 右图：信息质量累积曲线 (Cumulative Mass Curve) - 你的论文神图
    axes[1].plot(token_pct, cumulative_mass_pct, color='darkorange', linewidth=3, label='Cumulative Info Mass')
    
    # 画出 95% 达标参考线
    axes[1].axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='95% Mass Target')
    axes[1].axvline(x=tokens_needed_95, color='green', linestyle='--', linewidth=1.5, 
                    label=f'Tokens Needed: {tokens_needed_95:.1f}%')
    
    # 画出交点
    axes[1].scatter([tokens_needed_95], [95], color='red', s=100, zorder=5)
    
    axes[1].set_title(f'Image {img_idx} - Information Mass Preservation', fontsize=12)
    axes[1].set_xlabel('Percentage of Retained Tokens (%) - [Compute Cost]')
    axes[1].set_ylabel('Percentage of Retained Info Mass (%)')
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc='lower right')
    axes[1].grid(linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 保存图像
    save_dir = Path(__file__).parent / 'kl_scores_export'
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'image_{img_idx:02d}_mass_preservation.png'
    plt.savefig(save_path, dpi=150)
    print(f"  Saved plot to {save_path}")
    plt.close()