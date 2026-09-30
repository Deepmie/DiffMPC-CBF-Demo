import numpy as np
from diffmpc_cbf.solver import SQP
from diffmpc_cbf.utils import plot_trajectory

if __name__ == '__main__':
    x0   = np.array([[0.], [0.]])
    xref = np.array([[1.], [0.]])
    sqp  = SQP()
    xs   = sqp.solve(x0, xref, total_step=50)
    plot_trajectory(xs, 'imgs/sqp.png')
