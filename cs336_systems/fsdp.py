import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.model import Linear, Embedding

class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.handles = []
        self.rank = dist.get_rank()
        self._broadcast_params()
        self._shard_params()
        self._register_hooks()
    def _broadcast_params(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _shard_params(self):
        for mod in self.module.modules():
            if isinstance(mod, (Linear, Embedding)):
                chunks = torch.chunk(mod.weight.data, self.world_size, dim=0)
                shard = chunks[self.rank] 
                mod.weight.data = shard 

    def _register_hooks(self):
        for mod in self.module.modules():
            if isinstance(mod, (Linear, Embedding)):
                mod.register_forward_pre_hook(self._make_forward_pre_hook(mod))
                mod.register_forward_hook(self._make_forward_post_hook(mod))
            if isinstance(mod, Linear):
                mod.register_full_backward_pre_hook(self._make_backward_pre_hook(mod))
            for param in mod.parameters(recurse=False):
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_grad_hook(mod))

    def _make_forward_pre_hook(self, mod):
        def hook(mod, inp):
            l = [torch.zeros_like(mod.weight.data) for _ in range(self.world_size)]
            dist.all_gather(l, mod.weight.data)
            mod._saved_shard = mod.weight.data
            full_weight = torch.cat(l, dim=0)
            if self.compute_dtype is not None:
                full_weight = full_weight.to(self.compute_dtype)
            mod.weight.data = full_weight
        return hook
    
    def _make_backward_pre_hook(self, mod):
        def hook(mod, grad_output):
            l = [torch.zeros_like(mod.weight.data) for _ in range(self.world_size)]
            dist.all_gather(l, mod.weight.data)
            mod._saved_shard = mod.weight.data
            full_weight = torch.cat(l, dim=0)
            if self.compute_dtype is not None:
                full_weight = full_weight.to(self.compute_dtype)
            mod.weight.data = full_weight
        return hook
        
    def _make_backward_post_hook(self, mod):
        def hook(mod, inp, out):
            mod.weight.data = mod._saved_shard
            del mod._saved_shard
        return hook
    
    def _make_forward_post_hook(self, mod):
        def hook(mod, inp, out):
            mod.weight.data = mod._saved_shard
            del mod._saved_shard
        return hook
    
    def _make_grad_hook(self, mod):
        if isinstance(mod, (Linear, Embedding)):
            def hook(param):
                grad = param.grad
                if hasattr(mod, '_saved_shard'):
                    mod.weight.data = mod._saved_shard
                    del mod._saved_shard
                grad = grad.to(mod.weight.dtype)
                grad /= self.world_size
                chunks = list(torch.chunk(grad, self.world_size, dim=0))
                output = torch.empty_like(chunks[self.rank])
                handle = dist.reduce_scatter(output, chunks, async_op=True)
                param.grad = output
                self.handles.append(handle)
            return hook
        else:
            def hook(param):
                param.grad /= self.world_size
                handle = dist.all_reduce(param.grad, async_op=True)
                self.handles.append(handle)
            return hook
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        res = {}
        for name, mod in self.module.named_modules():
            for pname, param in mod.named_parameters(recurse=False):
                if name == "":
                    key = pname
                else:
                    key = f"{name}.{pname}"

                if isinstance(mod, (Linear, Embedding)):
                    l = [torch.zeros_like(mod.weight.data) for _ in range(self.world_size)] 
                    dist.all_gather(l, mod.weight.data)
                    value = torch.cat(l, dim=0)
                else:
                    value = param.data.clone()

                res[key] = value
        return res