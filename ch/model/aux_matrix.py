import numpy as np
import scipy
import scipy.sparse as sp
from .utils import check_random_state, get_args

conf = get_args()
N = conf.N
T = conf.T
init = conf.init
seed = conf.seed
mu = conf.mu


def init_mat():
    c_vecs = np.zeros((T, (N + 1)**2))
    if init == 1:
        rng = check_random_state(seed)
        c_vecs[0] = 2*rng.rand((N+1)**2) - 1
    elif init == 2:
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        xx, yy = np.meshgrid(x, y)
        c_mat = np.cos(2*np.pi*xx)*np.cos(np.pi*yy)
        c_vecs[0] = c_mat.flatten('F')
    return c_vecs


def generate_2d():
    """
    Generate components matrix to solve in difference schemes.
    :return P: Sparse (N+1)**2 Identity Matirx
    :return Q: -mu*D
    :return S: negative P
    :return D: Laplacian operator
    """
    P = sp.eye((N+1)**2)
    S = -1*sp.eye((N+1)**2)
    fd1D = sp.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1))
    I = sp.eye(N+1)
    D = sp.kron(I, fd1D) + sp.kron(fd1D, I)
    for i in range(N+1):
        D[i, i+1+N] = 2
        D[(i-1)*(N+1)+1, (i-1)*(N+1)+2] = 2
        D[(i-1)*(N+1)+N+1, (i-1)*(N+1)+N] = 2
    Q = -1*mu*D
    return P, Q, S, D
