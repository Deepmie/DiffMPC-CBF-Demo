import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from typing import List, Optional, Any

# Plot Tools
def plot_trajectory(xs: ndarray, save_path: Optional[str]=None): # (N, nx)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs[:, 0], xs[:, 1], marker='x', color='k')
    ax.set_xlabel('p'); ax.set_ylabel('v')
    ax.set_title('Trajectory')

    if save_path is not None: fig.savefig(save_path)
    else: plt.show()

# Convert Tools
def convert_tensor_to_numpy(t: Tensor) -> ndarray:
    return t.cpu().detach().numpy().flatten()

def convert_numpy_to_tensor(n: ndarray) -> Tensor:
    return torch.from_numpy(n).requires_grad_()

def convert_tensor_to_numpy_batch(ts: List[Tensor]) -> List[ndarray]:
    return [convert_tensor_to_numpy(t) for t in ts]

def convert_numpy_to_tensor_batch(ns: List[ndarray]) -> List[Tensor]:
    return [convert_numpy_to_tensor(n) for n in ns]

def convert_tensor_iteration(root: Any, device: torch.device) -> Any:
    if isinstance(root, (str, Tensor)):
        return root
    elif isinstance(root, (int, float)):
        return torch.tensor(root, device=device)
    elif isinstance(root, ndarray):
        return torch.from_numpy(root.copy()).to(device)
    elif isinstance(root, list):
        return [convert_tensor_iteration(e, device) for e in root]
    elif isinstance(root, dict):
        return {k: convert_tensor_iteration(v, device) for k, v in root.items()}
    else:
        raise TypeError(f'type of root is {type(root)} and not in support list: [str, int, float, ndarray, list, dict]')
