import torch
import triton.language as tl
import triton

class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        B, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        Br, Bc = 32, 32
        o = torch.zeros_like(q)
        lse = torch.zeros(B, N, device=q.device, dtype=q.dtype)
        
        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]            # (B, Br, D)
            o_curr = torch.zeros(B, i_end - i, D, device=q.device, dtype=q.dtype)
            l_curr = torch.zeros(B, i_end - i, 1, device=q.device, dtype=q.dtype)
            m_curr = torch.full((B, i_end - i, 1), float('-inf'), device=q.device, dtype=q.dtype)
            
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                Vj = v[:, j:j_end]
                
                S = Qi @ Kj.transpose(-2, -1) * scale
                
                if is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]   # (Br, 1)
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]   # (1, Bc)
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                
                S_max = S.amax(dim=-1, keepdim=True) 
                m_new = torch.maximum(m_curr, S_max)
                o_curr = o_curr * torch.exp(m_curr - m_new)
                l_curr = l_curr * torch.exp(m_curr - m_new)
                P = torch.exp(S - m_new)
                o_curr = o_curr + P @ Vj
                l_curr = l_curr + P.sum(-1, keepdim=True)
                m_curr = m_new
            
            o[:, i:i_end] = o_curr / l_curr
            lse[:, i:i_end] = (m_curr + torch.log(l_curr)).squeeze(-1)
        
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, lse = ctx.saved_tensors     # (B, N)
        # q, k, v 需要在 forward 里也 save_for_backward
        # 否则 backward 拿不到它们
        
        B, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        Br, Bc = 32, 32
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # 第一遍：重算每个 query 块的 m
        m_all = torch.full((B, N), float('-inf'), device=q.device)
        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]
            m_i = torch.full((B, i_end - i), float('-inf'), device=q.device)
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                S = Qi @ Kj.transpose(-2, -1) * scale
                if ctx.is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]   # (Br, 1)
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]   # (1, Bc)
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                m_i = torch.maximum(m_i, S.amax(dim=-1))
            m_all[:, i:i_end] = m_i
        
        D = torch.zeros(B, N, device=q.device, dtype=q.dtype)
        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]
            Di = do[:, i:i_end]
            lse_i = lse[:, i:i_end, None]
            D_i = torch.zeros(B, i_end - i, device=q.device, dtype=q.dtype)
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                Vj = v[:, j:j_end]
                S = Qi @ Kj.transpose(-2, -1) * scale
                if ctx.is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                P = torch.exp(S - lse_i)
                D_i += (Di @ Vj.transpose(-2, -1) * P).sum(-1)
            D[:, i:i_end] = D_i

        # 第二遍：用 m 和 lse 算梯度
        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]
            dQ_i = torch.zeros_like(Qi)
            m_i = m_all[:, i:i_end, None]          # (B, Br, 1)
            lse_i = lse[:, i:i_end, None]           # (B, Br, 1)
            Di = do[:, i:i_end]                     # (B, Br, D)
            
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                Vj = v[:, j:j_end]
                S = Qi @ Kj.transpose(-2, -1) * scale
                if ctx.is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]   # (Br, 1)
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]   # (1, Bc)
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                
                # P = exp(S - m) / ℓ = exp(S - lse)
                P = torch.exp(S - lse_i)            # (B, Br, Bc)
                
                # softmax 梯度
                dP = P * (Di @ Vj.transpose(-2, -1) - D[:, i:i_end, None])
                
                dQ_i += dP @ Kj * scale
                dk[:, j:j_end] += dP.transpose(-2, -1) @ Qi * scale 
                dv[:, j:j_end] += P.transpose(-2, -1) @ Di
            
            dq[:, i:i_end] = dQ_i
        
        return dq, dk, dv, None

@triton.jit
def _flash_forward_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    N,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lseb, stride_lsen,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_block_idx = tl.program_id(1)
    q_start = q_block_idx * BLOCK_Q

    q_rows = q_start + tl.arange(0, BLOCK_Q)
    q_cols = tl.arange(0, D)
    q_offsets = batch_idx * stride_qb + q_rows[:, None] * stride_qn + q_cols[None, :] * stride_qd
    q_mask = q_rows[:, None] < N
    Q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0).to(tl.float32)

    m = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    o = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    n_kv_blocks = tl.cdiv(N, BLOCK_KV)
    for j in range(n_kv_blocks):
        kv_start = j * BLOCK_KV
        kv_rows = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask_k = (kv_rows < N)[None, :]
        kv_mask_v = (kv_rows < N)[:, None]

        k_offsets = batch_idx * stride_kb + kv_rows[None, :] * stride_kn + tl.arange(0, D)[:, None] * stride_kd
        K = tl.load(k_ptr + k_offsets, mask=kv_mask_k, other=0.0).to(tl.float32)

        v_offsets = batch_idx * stride_vb + kv_rows[:, None] * stride_vn + tl.arange(0, D)[None, :] * stride_vd
        V = tl.load(v_ptr + v_offsets, mask=kv_mask_v, other=0.0).to(tl.float32)

        S = tl.dot(Q, K) * scale

        if is_causal:
            row_idx = q_start + tl.arange(0, BLOCK_Q)[:, None]
            col_idx = kv_start + tl.arange(0, BLOCK_KV)[None, :]
            S = tl.where(row_idx >= col_idx, S, float('-inf'))

        S = tl.where((kv_rows < N)[None, :], S, float('-inf'))

        m_new = tl.maximum(m, tl.max(S, axis=1))
        scale_factor = tl.exp(m - m_new)
        o = o * scale_factor[:, None]
        l = l * scale_factor
        P = tl.exp(S - m_new[:, None])
        o = o + tl.dot(P, V)
        l = l + tl.sum(P, axis=1)
        m = m_new

    o = o / l[:, None]
    LSE = m + tl.log(l)

    o_offsets = batch_idx * stride_ob + q_rows[:, None] * stride_on + tl.arange(0, D)[None, :] * stride_od
    tl.store(o_ptr + o_offsets, o, mask=q_mask)

    lse_offsets = batch_idx * stride_lseb + q_rows * stride_lsen
    tl.store(lse_ptr + lse_offsets, LSE, mask=(q_rows < N))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        B, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        BLOCK_Q, BLOCK_KV = 32, 32

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = torch.zeros_like(q)
        lse = torch.zeros(B, N, device=q.device, dtype=torch.float32)

        grid = (B, triton.cdiv(N, BLOCK_Q))
        _flash_forward_kernel[grid](
            q, k, v, o, lse,
            N,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1),
            scale,
            is_causal,
            D=D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, lse = ctx.saved_tensors
        B, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        Br, Bc = 32, 32
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        m_all = torch.full((B, N), float('-inf'), device=q.device)
        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]
            m_i = torch.full((B, i_end - i), float('-inf'), device=q.device)
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                S = Qi @ Kj.transpose(-2, -1) * scale
                if ctx.is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                m_i = torch.maximum(m_i, S.amax(dim=-1))
            m_all[:, i:i_end] = m_i

        D = torch.zeros(B, N, device=q.device, dtype=q.dtype)
        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]
            Di = do[:, i:i_end]
            lse_i = lse[:, i:i_end, None]
            D_i = torch.zeros(B, i_end - i, device=q.device, dtype=q.dtype)
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                Vj = v[:, j:j_end]
                S = Qi @ Kj.transpose(-2, -1) * scale
                if ctx.is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                P = torch.exp(S - lse_i)
                D_i += (Di @ Vj.transpose(-2, -1) * P).sum(-1)
            D[:, i:i_end] = D_i

        for i in range(0, N, Br):
            i_end = min(i + Br, N)
            Qi = q[:, i:i_end]
            dQ_i = torch.zeros_like(Qi)
            lse_i = lse[:, i:i_end, None]
            Di = do[:, i:i_end]
            D_i_2d = D[:, i:i_end, None]
            for j in range(0, N, Bc):
                j_end = min(j + Bc, N)
                Kj = k[:, j:j_end]
                Vj = v[:, j:j_end]
                S = Qi @ Kj.transpose(-2, -1) * scale
                if ctx.is_causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    S[((row_idx < col_idx)[None, :, :]).expand_as(S)] = float('-inf')
                P = torch.exp(S - lse_i)
                dP = P * (Di @ Vj.transpose(-2, -1) - D_i_2d)
                dQ_i += dP @ Kj * scale
                dk[:, j:j_end] += dP.transpose(-2, -1) @ Qi * scale
                dv[:, j:j_end] += P.transpose(-2, -1) @ Di
            dq[:, i:i_end] = dQ_i

        return dq, dk, dv, None