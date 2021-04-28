import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from .utils import get_args
conf = get_args()
N = conf.N


def plot_git(c_vecs, ex_name):
    directory = './pics/%s' % ex_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    cnt = 0
    for c in c_vecs:
        cnt += 1
        i = c.reshape(N+1, N+1)
        plt.matshow(i)
        plt.colorbar()
        plt.savefig(directory+'/%03d.png' % cnt)
        plt.close()
    with imageio.get_writer(directory+'/%s.gif' % ex_name, mode='I') as writer:
        for filename in [directory+'/%03d.png' % i for i in np.arange(1, 200, 4)]:
            image = imageio.imread(filename)
            writer.append_data(image)
