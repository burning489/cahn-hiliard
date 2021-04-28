import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from .aux_matrix import init_mat, generate_2d
from .utils import get_args
import warnings
warnings.filterwarnings("ignore")


conf = get_args()
eps = conf.eps
N = conf.N
T = conf.T
mu = conf.mu
max_iter = conf.max_iter
tol = conf.tol
H = 1/N


def ch2d11():
    c_vecs = init_mat()
    P, Q, S, D = generate_2d()
    print('diff scheme 1')
    for n in range(1, T):
        print('time step: {}'.format(n+1), end='\r')
        c_old = c_vecs[n-1]
        R = np.diag(c_old**2) - (eps ** 2/H ** 2)*D
        A1 = sp.hstack([P, Q])
        A2 = sp.hstack([R, S])
        A = sp.vstack([A1, A2])
        b = np.concatenate((c_old, c_old))
        x = spsolve(A, b)
        c_vecs[n] = x[:(N+1)**2]
    print(' '*20)
    return c_vecs


def ch2d12():
    c_vecs = init_mat()
    P, Q, S, D = generate_2d()
    R = 2*sp.eye((N+1)**2) - (eps ** 2/H ** 2)*D
    print('diff scheme 2')
    for n in range(1, T):
        print('time step: {}'.format(n+1), end='\r')
        c_old = c_vecs[n-1]
        A1 = sp.hstack([P, Q])
        A2 = sp.hstack([R, S])
        A = sp.vstack([A1, A2])
        b = np.concatenate((c_old, 3*c_old-np.power(c_old, 3)))
        x = spsolve(A, b)
        c_vecs[n] = x[:(N+1)**2]
    print(' '*20)
    return c_vecs


def ch2d13():
    c_vecs = init_mat()
    P, Q, S, D = generate_2d()
    print('diff scheme 3')
    for n in range(1, T):
        print('time step: {}'.format(n+1), end='\r')
        c_old = c_vecs[n-1]
        R = 3*np.diag(c_old**2) - np.identity((N+1)**2) - (eps ** 2/H ** 2)*D
        A1 = sp.hstack([P, Q])
        A2 = sp.hstack([R, S])
        A = sp.vstack([A1, A2])
        b = np.concatenate((c_old, 2*c_old**3))
        x = spsolve(A, b)
        c_vecs[n] = x[:(N+1)**2]
    print(' '*20)
    return c_vecs


def ch2d14():
    c_vecs = init_mat()
    w_vecs = np.zeros((T, (N + 1)**2))
    _, _, _, D = generate_2d()

    def g(c_new, c_old, w_new):
        f1 = c_new-c_old-mu*D*w_new
        f2 = w_new-c_new**3+c_old + (eps ** 2/H ** 2)*D*c_new
        return np.concatenate((f1, f2))

    for n in range(1, T):
        print('\rtime: {}{}'.format(n, " "*30))
        c_old = c_vecs[n-1]
        c_new = c_old
        w_old = w_vecs[n-1]
        w_new = w_old

        n_iter = 0
        err = tol+1
        while err > tol and n_iter < max_iter:
            print("\rn_iter:{} {}".format(n_iter+1, "*"*(n_iter+1)), end="")
            Dn = (eps ** 2/H ** 2)*D - np.diag(3*c_new ** 2)
            J1 = sp.hstack([sp.eye((N+1)**2), -mu*D])
            J2 = sp.hstack([Dn, sp.eye((N+1)**2)])
            J = sp.vstack([J1, J2])
            gn = g(c_new, c_old, w_new)
            s = -1 * spsolve(J, gn)
            c_new_new = c_new + s[:(N+1)**2]
            w_new_new = w_new + s[(N+1)**2:]
            err = np.linalg.norm(s-np.concatenate((c_new, w_new)), np.inf)
            c_new, w_new = c_new_new, w_new_new
            n_iter += 1
        c_vecs[n] = c_new
        w_vecs[n] = w_new
    return c_vecs


def ch2d15():
    c_vecs = init_mat()
    w_vecs = np.zeros((T, (N + 1)**2))
    _, _, _, D = generate_2d()

    def g(c_new, c_old, w_new):
        f1 = c_new-c_old-mu*D*w_new
        f2 = w_new-c_new**3+c_new + (eps ** 2/H ** 2)*D*c_new
        return np.concatenate((f1, f2))

    for n in range(1, T):
        print('\rtime: {}{}'.format(n, " "*30))
        c_old = c_vecs[n-1]
        c_new = c_old
        w_old = w_vecs[n-1]
        w_new = w_old

        n_iter = 0
        err = tol+1
        while err > tol and n_iter < max_iter:
            print("\rn_iter:{} {}".format(n_iter+1, "*"*(n_iter+1)), end="")
            Dn = (eps ** 2/H ** 2)*D - np.diag(3*c_new ** 2) + sp.eye((N+1)**2)
            J1 = sp.hstack([sp.eye((N+1)**2), -mu*D])
            J2 = sp.hstack([Dn, sp.eye((N+1)**2)])
            J = sp.vstack([J1, J2])
            gn = g(c_new, c_old, w_new)
            s = -1 * spsolve(J, gn)
            c_new_new = c_new + s[:(N+1)**2]
            w_new_new = w_new + s[(N+1)**2:]
            err = np.linalg.norm(s-np.concatenate((c_new, w_new)), np.inf)
            c_new, w_new = c_new_new, w_new_new
            n_iter += 1
        c_vecs[n] = c_new
        w_vecs[n] = w_new
    return c_vecs
