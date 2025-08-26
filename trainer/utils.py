import torch

class CheckpointStrategy:
    def __init__(self, config):
        self.config = config

    def save_checkpoint(self, model, optimizer, loss, step, is_master):
        raise NotImplementedError("This method should be implemented by subclasses")

    def load_checkpoint(self, model, optimizer, device, is_master):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    
def _to_scalar(x):
        try:
            from torch.distributed._tensor import DTensor  # PyTorch 2.7+
            if isinstance(x, DTensor):
                return x.to_local().detach().float().item()
        except Exception:
            pass
        if isinstance(x, torch.Tensor):
            return x.detach().float().item()
        return float(x)