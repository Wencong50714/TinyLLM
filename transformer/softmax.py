import torch
import triton
import triton.language as tl

from .utils import calc_num_warps
from pathlib import Path

@triton.jit
def _softmax_kernel_fused(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):
    # setup input ptrs
    row_index = tl.program_id(0)

    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols

    # move to SRAM
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    # softmax itself
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # write back to HBM
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton impl of Softmax, fwd pass only"""
    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now"
    block_size = triton.next_power_of_2(cols)
    num_warps = calc_num_warps(block_size)

    grid = (rows,)

    # allocate our output buffer
    sm_out = torch.empty_like(x)

    _softmax_kernel_fused[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )

    return sm_out

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            1024 * i for i in range(2, 50)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "fused",
            "torch",
        ],  # possible values for `line_arg``
        line_names=[
            "fused (Triton)",
            "torch",
        ],  # label name for the lines
        styles=[
            ("blue", "-"),
            ("green", "-"),
        ],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "fused":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_softmax(x), quantiles=quantiles
        )
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=-1), quantiles=quantiles
        )

    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())