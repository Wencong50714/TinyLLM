import matplotlib.pyplot as plt
import numpy as np
import math

def get_lr(it, learning_rate=5e-4, warmup_iters=1000, lr_decay_iters=100000, min_lr=0.0):
    """
    根据当前的训练迭代步数 it 返回当前的学习率值。
    学习率调整策略包括线性预热、余弦退火和最小学习率限制。
    """
    # 1) 线性预热阶段，在 warmup_iters 之前，学习率线性增加到目标学习率
    if it < warmup_iters:
        return learning_rate * it / warmup_iters  # 预热阶段，学习率线性增长

    # 2) 如果迭代步数超过 lr_decay_iters，返回最小学习率 min_lr
    if it > lr_decay_iters:
        return min_lr  # 训练进入尾声时，学习率达到最小值并保持不变

    # 3) 余弦退火阶段，在 warmup_iters 和 lr_decay_iters 之间，学习率逐渐降低
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1  # 确保衰减比在合法范围内
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦函数计算衰减系数，范围为0到1
    return min_lr + coeff * (learning_rate - min_lr)  # 根据衰减系数调整学习率

def plot_lr_schedule(ax=None):
    # 从train.py中获取的默认参数
    learning_rate = 5e-4
    max_iters = 100000
    warmup_iters = 1000
    lr_decay_iters = max_iters
    min_lr = 5e-5  # 最小学习率
    
    # 生成迭代步数点
    iterations = np.arange(0, max_iters + 5000, 100)  # 从0到max_iters+1000，步长100
    
    # 计算每个步数对应的学习率
    learning_rates = [get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr) for it in iterations]
    
    # 使用提供的 ax 或创建新的图表
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制学习率曲线
    ax.plot(iterations, learning_rates, 'b-', linewidth=2)
    
    # 标记重要区域
    ax.axvline(x=warmup_iters, color='r', linestyle='--', alpha=0.7, label=f'End of Warmup ({warmup_iters})')
    ax.axvline(x=lr_decay_iters, color='g', linestyle='--', alpha=0.7, label=f'End of Decay ({lr_decay_iters})')
    
    # 设置图表标题和轴标签
    ax.set_title('Learning Rate Schedule Visualization', fontsize=16)
    ax.set_xlabel('Training Iterations', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 标记不同阶段
    ax.annotate('Warmup Phase\n(Linear Growth)', xy=(warmup_iters/2, learning_rate/2), 
                 xytext=(warmup_iters/2, learning_rate*0.7), ha='center', fontsize=12,
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6))
    
    mid_decay = (warmup_iters + lr_decay_iters) / 2
    ax.annotate('Cosine Decay Phase', xy=(mid_decay, get_lr(mid_decay, learning_rate, warmup_iters, lr_decay_iters, min_lr)), 
                 xytext=(mid_decay, learning_rate*0.7), ha='center', fontsize=12,
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6))
    
    if max_iters < lr_decay_iters:
        ax.set_xlim(0, lr_decay_iters + 1000)
    
    return ax

def plot_cosine_decay_wave_details(ax=None):
    """
    可视化余弦退火的波形细节和峰值递减特性。
    此函数使用余弦波形的基本原理绘制多个周期，展示其振幅如何逐渐降低。
    """
    # 获取默认参数
    learning_rate = 5e-4
    min_lr = 5e-5
    
    # 创建一个演示性的余弦波形
    # 使用比实际训练更短的周期来清晰展示波形特征
    periods = 5  # 展示5个完整周期
    points_per_period = 100
    total_points = periods * points_per_period
    
    # 创建x轴（可以理解为迭代次数的抽象表示）
    x = np.linspace(0, periods * np.pi, total_points)
    
    # 计算余弦波的包络线 - 表示峰值递减
    envelope = np.linspace(learning_rate, min_lr, total_points)
    
    # 计算余弦波
    # 使用更高的频率来展示多个周期
    frequency = 2  # 控制波形的频率
    wave = 0.5 * (1 + np.cos(frequency * x))  # 基础余弦波，值范围在0-1之间
    
    # 计算递减的余弦波 - 将波形调整到学习率范围内，且整体递减
    decaying_wave = min_lr + wave * (envelope - min_lr)
    
    # 使用提供的 ax 或创建新的图表
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制递减的余弦波
    ax.plot(x, decaying_wave, 'b-', linewidth=2, label='Decaying Cosine Wave')
    
    # 绘制包络线（峰值递减曲线）
    ax.plot(x, envelope, 'r--', linewidth=1.5, label='Upper Envelope (Peak Decay)')
    
    # 绘制最小值包络线
    min_envelope = min_lr + (envelope - min_lr) * 0  # 最小包络线
    ax.plot(x, min_envelope, 'g--', linewidth=1.5, label='Lower Envelope (min_lr)')
    
    # 标记波峰
    peaks = []
    for i in range(periods):
        peak_x = i * np.pi / frequency
        peak_index = int(peak_x / (periods * np.pi) * total_points)
        if peak_index < total_points:
            peak_y = decaying_wave[peak_index]
            peaks.append((peak_x, peak_y))
            ax.scatter(peak_x, peak_y, color='purple', s=100, zorder=3)
    
    # 连接所有波峰，更清晰地展示峰值递减
    if peaks:
        peak_x, peak_y = zip(*peaks)
        ax.plot(peak_x, peak_y, 'purple', linestyle='-.', linewidth=1.5, label='Peak Decay Trend')
    
    # 标注重要点
    for i, (peak_x, peak_y) in enumerate(peaks):
        ax.annotate(f'Peak {i+1}: {peak_y:.1e}', 
                    xy=(peak_x, peak_y),
                    xytext=(peak_x + 0.2, peak_y + 0.00005),
                    arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                    fontsize=10)
    
    # 添加阴影区域展示振幅递减
    for i in range(periods):
        if i < periods - 1:
            start_x = i * np.pi / frequency
            end_x = (i + 1) * np.pi / frequency
            start_idx = int(start_x / (periods * np.pi) * total_points)
            end_idx = int(end_x / (periods * np.pi) * total_points)
            
            x_section = x[start_idx:end_idx]
            y_upper = envelope[start_idx:end_idx]
            y_lower = min_envelope[start_idx:end_idx]
            
            ax.fill_between(x_section, y_lower, y_upper, alpha=0.1, color=f'C{i}')
    
    # 设置图表标题和轴标签
    ax.set_title('Cosine Decay Wave Pattern and Amplitude Reduction', fontsize=16)
    ax.set_xlabel('Phase (π units)', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    return ax

def plot_combined_lr_visualizations():
    """绘制组合学习率可视化图，水平拼接两个图表"""
    # 创建一个有两个子图的大图，水平排列
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    # 在左侧子图绘制学习率调度
    plot_lr_schedule(axs[0])
    
    # 在右侧子图绘制余弦衰减波形细节
    plot_cosine_decay_wave_details(axs[1])
    
    # 整体标题
    plt.suptitle('Learning Rate Schedule Comprehensive Visualization', fontsize=18, y=0.98)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('combined_lr_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 单独生成图像
    # plot_lr_schedule()
    # plot_cosine_decay_wave_details()
    
    # 生成组合图像
    plot_combined_lr_visualizations()
    print("Combined learning rate visualization generated and saved as 'combined_lr_visualization.png'")