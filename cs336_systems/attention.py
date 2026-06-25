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
        
        ctx.save_for_backward(lse)
        return o

    @staticmethod
    def backward(ctx, do):
        dq = 0
        dk = 0
        dv = 0
        
        return dq, dk, dv, None