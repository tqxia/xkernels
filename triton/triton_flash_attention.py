import torch
from torch import nn

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    o_block,
    l_i,
    m_i,
    q_block,
    k_block_ptr,
    v_block_ptr,
    block_index_q,
    softmax_scale,
    block_size_q: tl.constexpr,
    block_size_kv: tl.constexpr,
    stage: tl.constexpr,
    offsets_q: tl.constexpr,
    offsets_kv: tl.constexpr,
    seq_len: tl.constexpr,
):
    # [lo, hi)
    if stage == 1:
        # from 0 to the left of the diagonal
        lo, hi = 0, block_index_q * block_size_q
    elif stage == 2:
        # used only for the blocks in which there is a transition from non-masked to masked
        # (along the diagonal)
        lo, hi = block_index_q * block_size_q, (block_index_q + 1) * block_size_q
        lo = tl.multiple_of(lo, block_size_q)
    else:
        # only used for non-causal attention
        lo, hi = 0, seq_len

    k_block_ptr = tl.advance(k_block_ptr, (0, lo))
    v_block_ptr = tl.advance(v_block_ptr, (lo, 0))
    
    for start_kv in range(lo, hi, block_size_kv):
        start_kv = tl.multiple_of(start_kv, block_size_kv)

        k_block = tl.load(k_block_ptr)
        # shape: (block_size_q, block_size_kv)
        qk_block= tl.dot(q_block, k_block)

        if stage == 2:
            # reshape offsets_q: (block_size_q) -> (block_size_q, 1)
            # reshape offsets_kv: (block_size_kv) -> (1, block_size_kv)
            # broadcast offsets_q: (block_size_q, 1) -> (block_size_q, block_size_kv)
            mask = offsets_q[:, None] >= (start_kv + offsets_kv[None, :])
            # why not using float("-inf")?
            # -inf can create NaNs in max/exp/other numerical ops and may not be supported
            # or safe in Triton hardware ops.
            qk_block = qk_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            # tl.maximum: elementwise maximum; tl.max: maximum along a dimension (reduce)
            m_ij = tl.maximum(m_i, tl.max(qk_block, 1))
            q_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk_block, 1) * softmax_scale)
            qk_block = qk_block * softmax_scale - m_ij[:, None]

        # compute exp(qk_ij - m_ij)
        p_block = tl.math.exp(qk_block)
        # compute the sum by rows of p_block
        l_ij = tl.sum(p_block, 1)

        # compute the correction factor
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        v_block = tl.load(v_block_ptr)
        p_block = p_block.to(tl.float16)

        o_block = o_block * alpha[:, None]
        o_block = tl.dot(p_block, v_block, o_block)

        m_i = m_ij
        k_block_ptr = tl.advance(k_block_ptr, (0, block_size_kv))
        v_block_ptr = tl.advance(v_block_ptr, (block_size_kv, 0))


    return o_block, l_i, m_i


@triton.jit
def triton_attn(
    q,   # shape: (batch_size, num_heads, seq_len, head_dim)
    k,   # shape: (batch_size, num_heads, seq_len, head_dim)
    v,   # shape: (batch_size, num_heads, seq_len, head_dim)
    softmax_scale,
    m,   # logsumexp L in the original algorithm, shape: (batch_size, num_heads, seq_len)
    o,   # output, shape: (batch_size, num_heads, seq_len, head_dim)
    # strides
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_o_batch,
    stride_o_head,
    stride_o_seq,
    stride_o_dim,
    batch_size,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_size_q: tl.constexpr,  # Br
    block_size_kv: tl.constexpr, # Bc
    stage: tl.constexpr, # causal or non-causal
):
    # tl.static_assert(block_size_kv <= head_dim)
    # which block in the sequence length to process,
    # each of this triton kernel works with one query block
    block_index_q = tl.program_id(0)
    # which head of which batch to process.
    # each program is responsible for a single head of a single batch
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // num_heads
    index_head = index_batch_head % num_heads
    # get the (seq_len, head_dim) offset of the current batch and head
    qkv_offset = index_batch.to(tl.int64) * stride_q_batch + \
        index_head.to(tl.int64) * stride_q_head

    # "block pointer": a lightweight descriptor that maps a tile and its local
    # indices into concrete memory addresses.
    # block pointers can be indexed like tensors.

    # block q: [index_batch, index_head, block_index_q * block_size_q, :]
    q_block_ptr = tl.make_block_ptr(
        base=q + qkv_offset,
        shape=(seq_len, head_dim),
        strides=(stride_q_seq, stride_q_dim),
        offset=(block_index_q * block_size_q, 0),
        block_shape=(block_size_q, head_dim),
        order=(1, 0),
    )

    # block v: [index_batch, index_head, :, :]
    v_block_ptr = tl.make_block_ptr(
        base=v + qkv_offset,
        shape=(seq_len, head_dim),
        strides=(stride_v_seq, stride_v_dim),
        offset=(0, 0),
        block_shape=(block_size_kv, head_dim),
        order=(1, 0),
    )

    # block k: [index_batch, index_head, :, :], transposed k
    k_block_ptr = tl.make_block_ptr(
        base=k + qkv_offset,
        shape=(head_dim, seq_len),
        strides=(stride_k_dim, stride_k_seq),
        offset=(0, 0),
        block_shape=(head_dim, block_size_kv),
        order=(0, 1),
    )

    # block o: [index_batch, index_head, block_index_q * block_size_q, :]
    o_block_ptr = tl.make_block_ptr(
        base=o + qkv_offset,
        shape=(seq_len, head_dim),
        strides=(stride_o_seq, stride_o_dim),
        offset=(block_index_q * block_size_q, 0),
        block_shape=(block_size_q, head_dim),
        order=(1, 0),
    )

    # the offsets for each token in current q block
    offsets_q = block_index_q * block_size_q + tl.arange(0, block_size_q)
    # the offsets for each token in current k/v block
    offsets_kv = tl.arange(0, block_size_kv)
    # the running row maximum of a block of q * transpose(k)
    m_i = tl.zeros([block_size_q], dtype=tl.float32) - float("inf")
    # the ruuning sum, the normalization factor that be applied at the end
    l_i = tl.zeros([block_size_q], dtype=tl.float32) + 1.0
    o_block = tl.zeros([block_size_q, head_dim], dtype=tl.float32)
    # load q block from HBM to SRAM
    q_block = tl.load(q_block_ptr)

    if stage == 1 or stage == 3:
        # this branch runs for non-causal attention, 1 for non-causal and 3 for causal,
        # or for the blocks to the left of the diagonal of causal attention.
        o_block, l_i, m_i = _attn_fwd_inner(
            o_block,
            l_i,
            m_i,
            q_block,
            k_block_ptr,
            v_block_ptr,
            block_index_q,
            softmax_scale,
            block_size_q,
            block_size_kv,
            4 - stage,
            offsets_q,
            offsets_kv,
            seq_len,
        )

    if stage == 3:
        o_block, l_i, m_i = _attn_fwd_inner(
            o_block,
            l_i,
            m_i,
            q_block,
            k_block_ptr,
            v_block_ptr,
            block_index_q,
            softmax_scale,
            block_size_q,
            block_size_kv,
            2,
            offsets_q,
            offsets_kv,
            seq_len,
        )
    # needed for compute the logsumpexp for the backward pass
    m_i += tl.math.log(l_i)
    m_ptrs = m + index_batch_head * seq_len + offsets_q
    tl.store(m_ptrs, m_i)
    tl.store(o_block_ptr, o_block.to(o.type.element_ty))


class TritonAttention(nn.Module):
    def __init__(self, num_heads, head_dim):
        self.block_size_q = 128

    def forward(self, q, k, v):
        # input shape: (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = q.shape
        # number of parallel triton programs: batch_size * num_heads * block_size_q
        grid = lambda args: {
            # how many blocks of sequence length to process
            triton.cdiv(seq_len, self.block_size_q),
            # which head of which batch to process
            batch_size * num_heads,
            1,
        }
