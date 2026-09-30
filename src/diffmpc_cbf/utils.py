import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

def plot_trajectory(xs: ndarray, save_path: str): # (N, nx)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs[0, :], xs[1, :], marker='x', color='k')
    ax.set_xlabel('p'); ax.set_ylabel('v')
    ax.set_title('Trajectory')
    fig.savefig(save_path)