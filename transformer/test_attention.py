import torch
import time
import matplotlib.pyplot as plt

from utils import *
from torch_model import Attention as TorchAttention
from triton_model import TritonAttention


def test_attention():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 检查是否有可用的 CUDA 设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确保有可用的 GPU")

    # 创建模型参数
    args = ModelArgs()

    # 初始化两个注意力模型
    torch_attention = TorchAttention(args)
    triton_attention = TritonAttention(args)

    # 确保两个模型使用相同的权重
    triton_attention.load_state_dict(torch_attention.state_dict())

    # 将模型移动到 CUDA 并转换为 float16
    torch_attention = torch_attention.cuda().half()
    triton_attention = triton_attention.cuda().half()

    # 创建输入数据
    batch_size = 2
    seq_len = 64
    dim = args.dim
    x = torch.randn(batch_size, seq_len, dim, device="cuda").half()

    # 预计算旋转位置嵌入
    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, seq_len)
    freqs_cos = freqs_cos.cuda().half()
    freqs_sin = freqs_sin.cuda().half()

    # 设置模型为评估模式
    torch_attention.eval()
    triton_attention.eval()

    # 使用 torch.no_grad() 确保不计算梯度
    with torch.no_grad():
        # 获取两个模型的输出
        torch_output = torch_attention(x, freqs_cos, freqs_sin)
        triton_output = triton_attention(x, freqs_cos, freqs_sin)

    # 计算输出之间的差异
    max_diff = torch.max(torch.abs(torch_output - triton_output))
    mean_diff = torch.mean(torch.abs(torch_output - triton_output))

    print(f"最大差异: {max_diff.item():.6f}")
    print(f"平均差异: {mean_diff.item():.6f}")

    # 检查输出形状是否相同
    print(f"PyTorch Attention 输出形状: {torch_output.shape}")
    print(f"Triton Attention 输出形状: {triton_output.shape}")

    # 检查是否所有元素都接近（使用更宽松的误差范围，因为 float16 精度较低）
    is_close = torch.allclose(torch_output, triton_output, rtol=1e-2, atol=1e-2)
    print(f"输出是否在误差范围内相等: {is_close}")


def benchmark_attention(seq_lens=[64, 128, 256, 512, 1024, 2048, 4096]):
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 检查是否有可用的 CUDA 设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确保有可用的 GPU")

    # 创建模型参数
    args = ModelArgs()

    # 初始化两个注意力模型
    torch_attention = TorchAttention(args)
    triton_attention = TritonAttention(args)

    # 确保两个模型使用相同的权重
    triton_attention.load_state_dict(torch_attention.state_dict())

    # 将模型移动到 CUDA 并转换为 float16
    torch_attention = torch_attention.cuda().half()
    triton_attention = triton_attention.cuda().half()

    # 预热
    batch_size = 2
    dim = args.dim
    x = torch.randn(batch_size, 64, dim, device="cuda").half()
    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, 64)

    for _ in range(10):
        with torch.no_grad():
            torch_attention(x, freqs_cos, freqs_sin)
            triton_attention(x, freqs_cos, freqs_sin)

    # 性能测试
    torch_times = []
    triton_times = []
    torch_memory_peaks = []
    triton_memory_peaks = []

    for seq_len in seq_lens:
        print(f"\n测试序列长度: {seq_len}")

        # 创建输入数据
        x = torch.randn(batch_size, seq_len, dim, device="cuda").half()
        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, seq_len)

        # 测试 PyTorch 实现
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # 重置内存峰值统计
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                torch_attention(x, freqs_cos, freqs_sin)
        torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / 100 * 1000  # 转换为毫秒
        torch_times.append(torch_time)
        # 获取内存峰值使用
        torch_memory_peak = torch.cuda.max_memory_allocated() / (1024**2)  # 转换为MB
        torch_memory_peaks.append(torch_memory_peak)
        print(
            f"PyTorch Attention 平均时间: {torch_time:.2f} ms, 内存峰值: {torch_memory_peak:.2f} MB"
        )

        # 测试 Triton 实现
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # 重置内存峰值统计
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                triton_attention(x, freqs_cos, freqs_sin)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / 100 * 1000  # 转换为毫秒
        triton_times.append(triton_time)
        # 获取内存峰值使用
        triton_memory_peak = torch.cuda.max_memory_allocated() / (1024**2)  # 转换为MB
        triton_memory_peaks.append(triton_memory_peak)
        print(
            f"Triton Attention 平均时间: {triton_time:.2f} ms, 内存峰值: {triton_memory_peak:.2f} MB"
        )

    # 绘制性能对比图
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(seq_lens, torch_times, "o-", label="PyTorch Attention")
    plt.plot(seq_lens, triton_times, "o-", label="Triton Attention")
    plt.xlabel("Sequence Length")
    plt.ylabel("Computation Time (ms)")
    plt.title("Attention Performance Comparison")
    plt.grid(True)
    plt.legend()

    # 绘制内存峰值对比图
    plt.subplot(1, 2, 2)
    plt.plot(seq_lens, torch_memory_peaks, "o-", label="PyTorch Attention")
    plt.plot(seq_lens, triton_memory_peaks, "o-", label="Triton Attention")
    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Memory Usage Comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("attention_benchmark.png")
    plt.close()

    return torch_times, triton_times, torch_memory_peaks, triton_memory_peaks


if __name__ == "__main__":
    # 运行正确性测试
    test_attention()

    # 运行性能测试
    print("\n开始性能测试...")
    torch_times, triton_times, torch_memory_peaks, triton_memory_peaks = (
        benchmark_attention()
    )
