import numpy as np
from ..model.ch2d12 import ch2d12
from ..model.plot import plot_git


def ex1():
    c_vecs = ch2d12()
    np.savetxt('./c_vecs.txt', c_vecs)
    print(c_vecs[-1])
    plot_git(c_vecs, 'ex1')
