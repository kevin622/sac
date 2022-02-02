import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

def to_tensor(array: np.ndarray):
    return torch.from_numpy(array).to(device).to(torch.float32)

def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)