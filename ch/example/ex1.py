import numpy as np
from ..model.diff_scheme import ch2d11, ch2d12, ch2d13, ch2d14, ch2d15
from ..model.plot import plot_git


def ex1(n=5):
    simulations = [ch2d11, ch2d12, ch2d13, ch2d14, ch2d15]
    ex_names = ['ex{}'.format(i) for i in range(1, 6)]
    for simulation, ex_name in zip(simulations[:n], ex_names[:n]):
        c_vecs = simulation()
        # print('end state: ', c_vecs[-1])
        plot_git(c_vecs, ex_name)
