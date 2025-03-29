# reference https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rope_embedding.py

import triton
import triton.language as tl
import torch

from pathlib import Path

@triton.jit
def _fused_apply_rope_fwd(Q, K, COS, SIN,
                          B, COS_B, L, QH, KH, D:tl.constexpr, HALF_D:tl.constexpr,
                          BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr, BLOCK_D:tl.constexpr,
                          CHUNK_N:tl.constexpr=64
                          ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1)
    if start_n >= B*L:
        return
    off_b = start_n // L
    off_l = start_n % L

    Q += start_n * QH * D
    K += start_n * KH * D
    COS += off_b * (COS_B // B) * L * D + off_l * D
    SIN += off_b * (COS_B // B) * L * D + off_l * D

    cols = tl.arange(0, BLOCK_D)
    cos1 = tl.load(COS + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    cos2 = tl.load(COS + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]
    sin1 = tl.load(SIN + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    sin2 = tl.load(SIN + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]

    k1 = tl.load(K + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    k2 = tl.load(K + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    k_embed1 = k1 * cos1 - k2 * sin1
    k_embed2 = k2 * cos2 + k1 * sin2
    tl.store(K + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :], k_embed1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_KH)[:, None] < KH))
    tl.store(K + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :] + HALF_D, k_embed2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH))
    
    q1 = tl.load(Q + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    q2 = tl.load(Q + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    q_embed1 = q1 * cos1 - q2 * sin1
    q_embed2 = q2 * cos2 + q1 * sin2
    tl.store(Q + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :], q_embed1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_QH)[:, None] < QH))
    tl.store(Q + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :] + HALF_D, q_embed2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH))
    

@triton.jit
def _fused_apply_rope_bwd(DQ, DK, COS, SIN,
                          dq_stride_b, dq_stride_h, dq_stride_l, dq_stride_d,
                          dk_stride_b, dk_stride_h, dk_stride_l, dk_stride_d,
                        B, COS_B, L, QH, KH, D: tl.constexpr, HALF_D: tl.constexpr,
                        BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr, BLOCK_D:tl.constexpr,
                        CHUNK_N:tl.constexpr=64
                        ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1)
    if start_n >= B*L:
        return
    off_b = start_n // L
    off_l = start_n % L

    DQ += off_b * dq_stride_b + off_l * dq_stride_l
    DK += off_b * dk_stride_b + off_l * dk_stride_l
    COS += off_b * (COS_B // B) * L * D + off_l * D
    SIN += off_b * (COS_B // B) * L * D + off_l * D

    cols = tl.arange(0, BLOCK_D)
    cos1 = tl.load(COS + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    cos2 = tl.load(COS + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]
    sin1 = tl.load(SIN + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    sin2 = tl.load(SIN + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]

    dk_embed1 = tl.load(DK+ tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    dk_embed2 = tl.load(DK + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    dk1 = dk_embed1 * cos1 + sin2 * dk_embed2
    dk2 = dk_embed2 * cos2 - sin1 * dk_embed1
    tl.store(DK + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :], dk1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_KH)[:, None] < KH))
    tl.store(DK + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :] + HALF_D, dk2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH))

    dq_embed1 = tl.load(DQ + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    dq_embed2 = tl.load(DQ + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    dq1 = dq_embed1 * cos1 + sin2 * dq_embed2
    dq2 = dq_embed2 * cos2 - sin1 * dq_embed1
    tl.store(DQ + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :], dq1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_QH)[:, None] < QH))
    tl.store(DQ + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :] + HALF_D, dq2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH))

class _FusedApplyRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.transpose(1,2).is_contiguous() and k.transpose(1,2).is_contiguous()
        assert cos.is_contiguous() and sin.is_contiguous()
        # print(q.stride(), k.stride())
        B, QH, L, D = q.shape
        HALF_D = D // 2
        KH = k.size(1)
        # assert (D % 32 == 0) or (D % 64 == 0) or (D % 128 == 0)
        BLOCK_D = triton.next_power_of_2(HALF_D)
        num_stages=4
        num_warps=8

        # q_embed = torch.empty(B, L, QH, D, device=q.device, dtype=k.dtype)
        # k_embed = torch.empty(B, L, KH, D, device=q.device, dtype=k.dtype)
        
        N = B * L 
        COS_B = cos.shape[0]
        BLOCK_QH = triton.next_power_of_2(QH)
        BLOCK_KH = triton.next_power_of_2(KH)
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'])
        _fused_apply_rope_fwd[(grid)](q,k,cos, sin,
                                    B, COS_B, L, QH, KH, D, HALF_D,
                                    BLOCK_QH, BLOCK_KH, BLOCK_D,
                                    num_warps=num_warps, num_stages=num_stages

        )

        ctx.save_for_backward(cos, sin)
        ctx.infos = (B, QH, KH, L, D, HALF_D, N, COS_B, BLOCK_QH, BLOCK_KH, BLOCK_D)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        B, QH, KH, L, D, HALF_D, N, COS_B, BLOCK_QH, BLOCK_KH, BLOCK_D = ctx.infos
        cos,sin = ctx.saved_tensors
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'])
        _fused_apply_rope_bwd[grid](dq, dk, cos, sin,
                                    *dq.stride(), 
                                    *dk.stride(), 
                                    B, COS_B, L, QH, KH, D, HALF_D, BLOCK_QH, BLOCK_KH, BLOCK_D,
                                    num_warps=ctx.num_warps, num_stages=ctx.num_stages
                                    )
        return dq, dk, None, None

fused_apply_rope = _FusedApplyRope.apply

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

torch.cuda.empty_cache()
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(1, 32+1, 1)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['FusedApplyRope', 'ApplyRope'],  # possible values for `line_arg``
        line_names=[
            "FusedApplyRope",
            "ApplyRope",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="forward",  # name for the plot. Used also as a file name for saving the plot.
        args={'qh':32, 'kh':32, 'head_dim': 128, 'bs': 2}
        # args={'bs': 2, 'num_head': 32, 'rope_head_dim': 32, 
        #       'nope_head_dim': 64, 'kv_lora_rank': 256},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(bs, seq_len, head_dim,qh, kh, provider):
    device = torch.device('cuda')
    dtype = torch.float16
    q = torch.randn(bs, seq_len, qh, head_dim, device=device, dtype=dtype).transpose(1,2)
    k = torch.randn(bs, seq_len, kh, head_dim, device=device, dtype=dtype).transpose(1,2)
    cos = torch.randn(bs, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn_like(cos)
    # cos_unsloth = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    # sin_unsloth = torch.randn_like(cos)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if provider == 'FusedApplyRope':
        ms = triton.testing.do_bench(lambda: fused_apply_rope(q, k, cos, sin))
    if provider == 'ApplyRope':
        ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb(q, k, cos, sin))

    return ms * 1e3

def valid_test():
    device = 'cuda'
    dtype = torch.float32
    bs, seq_len, num_q_head, num_k_head, head_dim = 4, 128, 32, 1, 192
    q1 = torch.randn(bs, seq_len, num_q_head, head_dim, device=device, dtype=dtype).transpose(1,2)
    q1.requires_grad_(True)
    k1  = torch.randn(bs, seq_len, num_k_head, head_dim,device=device, dtype=dtype).transpose(1,2)
    k1.requires_grad_(True)
    q2 = torch.randn(bs, seq_len, num_q_head, head_dim, device=device, dtype=dtype).transpose(1,2)
    q2.data.copy_(q1.data)
    q2.requires_grad_(True)
    k2  = torch.randn(bs, seq_len, num_k_head, head_dim, device=device, dtype=dtype).transpose(1,2)
    k2.data.copy_(k1.data)
    k2.requires_grad_(True)
    cos = torch.randn(bs, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn_like(cos)

    dy = torch.rand_like(q1)

    if k1.grad is not None:
        k1.grad.zero_()
        q1.grad.zero_()
    q_embed1, k_embed1 = apply_rotary_pos_emb(q1, k1, cos, sin)
    print(q_embed1.stride())
    loss = (q_embed1 * repeat_kv(k_embed1, num_q_head//num_k_head))
    print(loss.stride())
    loss.sum().backward()

    if k2.grad is not None:
        k2.grad.zero_()
        q2.grad.zero_()
    q_embed2, k_embed2 = fused_apply_rope(q2, k2, cos, sin)
    print(q_embed2.stride())
    loss = (q_embed2 * repeat_kv(k_embed2, num_q_head//num_k_head))
    print(loss.stride())
    loss.sum().backward()

    print(torch.allclose(q_embed1, q_embed2, atol=1e-3), torch.allclose(k_embed1, k_embed2, atol=1e-3))
    print(torch.allclose(q1.grad, q2.grad, atol=1e-3), torch.allclose(k1.grad, k2.grad, atol=1e-3))

if __name__ == "__main__":
    valid_test()
    benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())