import numpy as np
from scipy.signal import convolve2d


conv_kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

ON = 1
OFF = 0


def get_density(grid):
    return np.count_nonzero(grid) / grid.size


def grid_to_number(grid):
    flatten_grid = grid.flatten()
    # numpy does not handle very big integers, therefore we have to declare the
    # np.arange array to be of type object in otder to use the python integers
    return flatten_grid.dot(2**np.arange(flatten_grid.size, dtype=object)[::-1])


def number_to_grid(n, h, w):
    return np.array(list(np.binary_repr(n, width=h*w)), dtype=int).reshape(h, w)


def is_periodic(states):
    # print(states)
    if len(states) > 1:
        previous = None
        for n in states:
            if previous == n:
                # print(states)
                return True
            previous = n
    return False


def init_grid(height, width, init_density):
    grid = np.random.choice([OFF, ON], size=(height, width), p=[
                            1 - init_density, init_density])
    return grid


def update_grid(grid):
    count_grid = convolve2d(grid, conv_kernel, "same")
    result_grid = np.zeros_like(grid)

    result_grid[(count_grid == 2) & (grid == ON)] = ON
    result_grid[(count_grid == 3)] = ON
    return result_grid
