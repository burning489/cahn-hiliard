import numpy as np
import scipy
import scipy.sparse as sp
from .utils import check_random_state
from .config import N, T, INIT_MODE, SEED, MU


def init_mat():
    c_vecs = np.zeros((T, (N + 1)**2))
    if INIT_MODE == 1:
        rng = check_random_state(SEED)
        c_vecs[0] = 2*rng.rand((N+1)**2) - 1
    elif INIT_MODE == 2:
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        xx, yy = np.meshgrid(x, y)
        c_mat = np.cos(2*np.pi*xx)*np.cos(np.pi*yy)
        c_vecs[0] = c_mat.flatten('F')
    return c_vecs


def generate_2d():
    P = sp.eye((N+1)**2)
    S = -1*sp.eye((N+1)**2)
    fd1D = sp.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1))
    I = sp.eye(N+1)
    D = sp.kron(I, fd1D) + sp.kron(fd1D, I)
    for i in range(N+1):
        D[i, i+1+N] = 2
        D[(i-1)*(N+1)+1, (i-1)*(N+1)+2] = 2
        D[(i-1)*(N+1)+N+1, (i-1)*(N+1)+N] = 2
    Q = -1*MU*D
    return P, Q, S, D
