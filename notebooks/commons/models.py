from torch import nn

def num_trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

