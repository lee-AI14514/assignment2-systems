import torch

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
                    S[row_idx < col_idx] = float('-inf')
                
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
                    S[row_idx < col_idx] = float('-inf')
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
                    S[row_idx < col_idx] = float('-inf')
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
                    S[row_idx < col_idx] = float('-inf')
                
                # P = exp(S - m) / ℓ = exp(S - lse)
                P = torch.exp(S - lse_i)            # (B, Br, Bc)
                
                # softmax 梯度
                dP = P * (Di @ Vj.transpose(-2, -1) - D[:, i:i_end, None])
                
                dQ_i += dP @ Kj * scale
                dk[:, j:j_end] += dP.transpose(-2, -1) @ Qi * scale 
                dv[:, j:j_end] += P.transpose(-2, -1) @ Di
            
            dq[:, i:i_end] = dQ_i
        
        return dq, dk, dv, None