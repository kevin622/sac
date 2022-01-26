import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

def to_tensor(array: np.ndarray):
    return torch.from_numpy(array).to(device).to(torch.float32)