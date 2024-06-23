import torch
import functools

class TorchDistributedZeroFirst:
    def __init__(self, rank=0):
        self.rank = rank
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    def __enter__(self):
        if self.world_size > 1 and torch.distributed.get_rank() != self.rank:
            torch.distributed.barrier()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_size > 1 and torch.distributed.get_rank() == self.rank:
            torch.distributed.barrier()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapped

# Usage as a context manager with custom rank
with TorchDistributedZeroFirst(rank=0):
    print(f"DEBUG START RUNNING ON RANK {torch.distributed.get_rank()}")
    # Code that should run first on the specified RANK
    print(f"DEBUG ENDING RUNNING ON RANK {torch.distributed.get_rank()}")

# Usage as a decorator with custom rank
@TorchDistributedZeroFirst(rank=0)
def my_function():
    print(f"DEBUG START RUNNING ON RANK {torch.distributed.get_rank()}")
    # Function code that should run first on the specified RANK
    print(f"DEBUG ENDING RUNNING ON RANK {torch.distributed.get_rank()}")

# Example function call
if torch.distributed.is_initialized():
    my_function()
