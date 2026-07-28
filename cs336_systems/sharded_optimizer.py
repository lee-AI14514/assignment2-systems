import torch
import torch.distributed as dist

class ShardedOptimizer:
    def __init__(self, params, optimizer_cls: type, **kwargs):
        seen = set()
        self.params = []
        for p in params:
            if id(p) in seen or p.requires_grad == False:
                continue
            seen.add(id(p))
            self.params.append(p)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.owned_params = []
        for index, value in enumerate(self.params):
            if index % self.world_size == self.rank:
                self.owned_params.append(value)
        self.optimizer = optimizer_cls(self.owned_params, **kwargs)

    def step(self, closure=None):
        self.optimizer.step()
        for index, param in enumerate(self.params):
            owner_rank = index % self.world_size
            dist.broadcast(param.data, src=owner_rank)
    
    def zero_grad(self, set_to_none=True):
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()