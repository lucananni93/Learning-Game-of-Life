import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.signal import convolve2d


ON = 1
OFF = 0
INITIAL_DENSITY = 0.3

HEIGHT = 10
WIDTH = 10

conv_kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])


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

    ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()


if __name__ == '__main__':
    main()