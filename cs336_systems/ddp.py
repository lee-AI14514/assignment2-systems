import torch
import torch.nn as nn
import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []  # 存异步 all-reduce 的 handle
        self._broadcast_params()
        self._register_hooks()

    def _broadcast_params(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
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
