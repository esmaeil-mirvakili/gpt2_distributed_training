import torch
import inspect


def configure_optimizers(model, weight_decay, learning_rate, device):
    # only the learnable params
    learnable_param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    # params that are 2D or of higher dimension get weight decay
    decay_params = [param for name, param in learnable_param_dict.items() if param.dim() >= 2]
    # no weight decay for biases, layer normal, ...
    no_decay_params = [param for name, param in learnable_param_dict.items() if param.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    size_decay_params = sum(param.numel() for param in decay_params)
    size_no_decay_params = sum(param.numel() for param in no_decay_params)
    print(f'Model has {len(decay_params)} params with decay, with the size of {size_decay_params}.')
    print(f'Model has {len(no_decay_params)} params without decay, with the size of {size_no_decay_params}.')
    # check if the pytorch AdamW has fused capability
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    used_fused = fused_available and 'cuda' in device
    print(f'Using fused AdamW: {used_fused}')
    return torch.optim.AdamW(optim_groups, fused=used_fused, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)