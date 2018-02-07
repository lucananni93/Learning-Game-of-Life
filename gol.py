import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


ON = 1
OFF = 0
INITIAL_DENSITY = 0.3

HEIGHT = 100
WIDTH = 100


def init_grid(height, width, init_density):
    grid = np.random.choice([OFF, ON], size=(height, width), p=[1 - init_density, init_density])
    return grid


def update_grid(grid):
    padded_grid = np.lib.pad(grid, (1,1), 'constant', constant_values=0)
    result_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            patch = padded_grid[i: i + 3, j: j + 3]
            updated_value = update_value(patch)
            result_grid[i, j] = updated_value
    return result_grid


def update_value(patch):
    value = patch[1, 1]
    n_others = patch.sum() - value
    if value == ON:
        if n_others < 2:
            return OFF      # death by isolation
        elif 2 <= n_others <= 3:
            return ON       # survival
        elif n_others > 3:
            return OFF      # death by overpopulation
        else:
            raise ValueError("n neighbours inconsistent")
    elif value == OFF:
        if n_others == 3:
            return ON       # birth by reproduction
        else:
            return OFF
    else:
        raise ValueError("Cell is in a unknown state")


def main():
    fig, ax = plt.subplots()
    grid = init_grid(HEIGHT, WIDTH, INITIAL_DENSITY)
    mat = ax.matshow(grid)

    def update(i):
        nonlocal grid
        updated_grid = update_grid(grid)
        mat.set_data(updated_grid)
        grid = updated_grid
        return [mat]

    ani = animation.FuncAnimation(fig, update, interval=50)
    plt.show()


if __name__ == '__main__':
    main()
