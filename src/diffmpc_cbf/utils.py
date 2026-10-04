import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from typing import List

# Plot Tools
def plot_trajectory(xs: ndarray, save_path: str): # (N, nx)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs[:, 0], xs[:, 1], marker='x', color='k')
    ax.set_xlabel('p'); ax.set_ylabel('v')
    ax.set_title('Trajectory')
    fig.savefig(save_path)

# Convert Tools
def convert_tensor_to_numpy(t: Tensor) -> ndarray:
    return t.cpu().detach().numpy().flatten()

def convert_numpy_to_tensor(n: ndarray) -> Tensor:
    return torch.from_numpy(n).requires_grad_()

def convert_tensor_to_numpy_batch(ts: List[Tensor]) -> List[ndarray]:
    return [convert_tensor_to_numpy(t) for t in ts]

def convert_numpy_to_tensor_batch(ns: List[ndarray]) -> List[Tensor]:
    return [convert_numpy_to_tensor(n) for n in ns]