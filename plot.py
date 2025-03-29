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

def plot_lr_schedule():
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
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制学习率曲线
    plt.plot(iterations, learning_rates, 'b-', linewidth=2)
    
    # 标记重要区域
    plt.axvline(x=warmup_iters, color='r', linestyle='--', alpha=0.7, label=f'End of Warmup ({warmup_iters})')
    plt.axvline(x=lr_decay_iters, color='g', linestyle='--', alpha=0.7, label=f'End of Decay ({lr_decay_iters})')
    
    # 设置图表标题和轴标签
    plt.title('Learning Rate Schedule Visualization', fontsize=16)
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 标记不同阶段
    plt.annotate('Warmup Phase\n(Linear Growth)', xy=(warmup_iters/2, learning_rate/2), 
                 xytext=(warmup_iters/2, learning_rate*0.7), ha='center', fontsize=12,
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6))
    
    mid_decay = (warmup_iters + lr_decay_iters) / 2
    plt.annotate('Cosine Decay Phase', xy=(mid_decay, get_lr(mid_decay, learning_rate, warmup_iters, lr_decay_iters, min_lr)), 
                 xytext=(mid_decay, learning_rate*0.7), ha='center', fontsize=12,
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6))
    
    if max_iters < lr_decay_iters:
        plt.xlim(0, lr_decay_iters + 1000)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('learning_rate_schedule.png', dpi=300)
    plt.show()

def plot_cosine_decay_detail():
    """
    Visualizes the details of the cosine decay phase of the learning rate schedule.
    This function creates a separate plot that focuses specifically on the cosine decay portion.
    """
    # Get default parameters from train.py
    learning_rate = 5e-4
    warmup_iters = 1000
    lr_decay_iters = 100000
    min_lr = 5e-5
    
    # Create more granular points specifically for the cosine decay phase
    decay_iterations = np.linspace(warmup_iters, lr_decay_iters, 1000)
    decay_learning_rates = [get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr) for it in decay_iterations]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot the cosine decay curve
    plt.plot(decay_iterations, decay_learning_rates, 'b-', linewidth=2.5)
    
    # Add a reference line for the max learning rate
    plt.axhline(y=learning_rate, color='r', linestyle='--', alpha=0.6, label='Max Learning Rate')
    
    # Add a reference line for the min learning rate
    plt.axhline(y=min_lr, color='g', linestyle='--', alpha=0.6, label='Min Learning Rate')
    
    # Mark key points on the decay curve
    quarter_point = warmup_iters + (lr_decay_iters - warmup_iters) * 0.25
    half_point = warmup_iters + (lr_decay_iters - warmup_iters) * 0.5
    three_quarter_point = warmup_iters + (lr_decay_iters - warmup_iters) * 0.75
    
    plt.scatter([warmup_iters, quarter_point, half_point, three_quarter_point, lr_decay_iters], 
                [get_lr(warmup_iters, learning_rate, warmup_iters, lr_decay_iters, min_lr),
                 get_lr(quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr),
                 get_lr(half_point, learning_rate, warmup_iters, lr_decay_iters, min_lr),
                 get_lr(three_quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr),
                 min_lr], 
                color='purple', s=80, zorder=3, alpha=0.8)
    
    # Add annotations for the key points
    plt.annotate(f'Start: {learning_rate:.1e}', 
                 xy=(warmup_iters, learning_rate), 
                 xytext=(warmup_iters-5000, learning_rate*1.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                 fontsize=10)
    
    plt.annotate(f'25%: {get_lr(quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr):.1e}', 
                 xy=(quarter_point, get_lr(quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr)),
                 xytext=(quarter_point-10000, get_lr(quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr)*1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                 fontsize=10)
    
    plt.annotate(f'50%: {get_lr(half_point, learning_rate, warmup_iters, lr_decay_iters, min_lr):.1e}', 
                 xy=(half_point, get_lr(half_point, learning_rate, warmup_iters, lr_decay_iters, min_lr)),
                 xytext=(half_point+5000, get_lr(half_point, learning_rate, warmup_iters, lr_decay_iters, min_lr)*1.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                 fontsize=10)
    
    plt.annotate(f'75%: {get_lr(three_quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr):.1e}', 
                 xy=(three_quarter_point, get_lr(three_quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr)),
                 xytext=(three_quarter_point-15000, get_lr(three_quarter_point, learning_rate, warmup_iters, lr_decay_iters, min_lr)*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                 fontsize=10)
    
    plt.annotate(f'End: {min_lr:.1e}', 
                 xy=(lr_decay_iters, min_lr),
                 xytext=(lr_decay_iters-15000, min_lr*0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                 fontsize=10)
    
    # Set up the plot
    plt.title('Detailed View of Cosine Decay Phase', fontsize=16)
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add mathematical formula for reference
    formula = r"$lr = min\_lr + 0.5 \cdot (1 + \cos(\pi \cdot \frac{iter - warmup\_iters}{decay\_iters - warmup\_iters})) \cdot (max\_lr - min\_lr)$"
    plt.figtext(0.5, 0.01, formula, ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig('cosine_decay_detail.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_lr_schedule()
    plot_cosine_decay_detail()  # Add this line to also plot the detailed cosine decay
    print("Learning rate schedule plots generated and saved as 'learning_rate_schedule.png' and 'cosine_decay_detail.png'")