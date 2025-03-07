import torch
import triton
import triton.language as tl
from .utils import calc_num_warps

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