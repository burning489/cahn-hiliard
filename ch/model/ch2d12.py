import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from .aux_matrix import init_mat, generate_2d
from .config import EP, N, T

H = 1/N


def ch2d12():
    c_vecs = init_mat()
    P, Q, S, D = generate_2d()
    R = 2*sp.eye((N+1)**2) - (EP ** 2/H ** 2)*D

    for n in range(1, T):
        c_old = c_vecs[n-1]
        c_old = np.array(c_old, dtype=np.float64)
        A1 = sp.hstack([P, Q])
        A2 = sp.hstack([R, S])
        A = sp.vstack([A1, A2])
        b = np.concatenate((c_old, 3*c_old-np.power(c_old, 3)))
        A = sp.csc_matrix(A)
        x = spsolve(A, b)
        c_vecs[n] = x[:(N+1)**2]
    return c_vecs
