import argparse
import numpy as np


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance
    :param seed: None, int or instance of RandomState
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    :returns: random number generator
    :raises ValueError: if param seed is of wrong type, raise error
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def get_args():
    """
    Get the command-line arguments
    :usage:
    >>> conf = get_args()
    >>> param = conf.name
    """
    parser = argparse.ArgumentParser(
        description='Get default configuration for Cahn-Hiliard simulation.')
    parser.add_argument('--mu', default=1, type=float,
                        help='dt/(dx)**2')
    parser.add_argument('--N', default=64, type=int,
                        help='space discretion')
    parser.add_argument('--T', default=200, type=int,
                        help='total time steps')
    parser.add_argument('--eps', default=1e-2, type=float,
                        help='material thickness')
    parser.add_argument('--init', default=1, type=int,
                        help='field init mode')
    parser.add_argument('--seed', default=10, type=int,
                        help='random seed')
    parser.add_argument('--max_iter', default=25, type=int,
                        help='max iter in Newton\'s method')
    parser.add_argument('--tol', default=1e-6, type=float,
                        help='tolerance in Newton\'s method')
    return parser.parse_args()
