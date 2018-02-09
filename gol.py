import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.signal import convolve2d


ON = 1
OFF = 0
INITIAL_DENSITY = 0.4

HEIGHT = 50
WIDTH = 50

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


def get_density(grid):
    return np.count_nonzero(grid) / grid.size


def main():
    fig, ax = plt.subplots(1, 2)
    grid = init_grid(HEIGHT, WIDTH, INITIAL_DENSITY)
    density = get_density(grid)
    mat = ax[0].matshow(grid, cmap=plt.cm.gray_r)
    densities = [density]
    line, = ax[1].plot([0], densities)

    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0, 50)

    def update(i):
        nonlocal grid, densities
        updated_grid = update_grid(grid)
        updated_density = get_density(updated_grid)
        densities.append(updated_density)
        n = len(densities)
        xmin, xmax = ax[1].get_xlim()
        if n >= xmax:
            ax[1].set_xlim(xmin, xmax*2)
            ax[1].figure.canvas.draw()
        mat.set_data(updated_grid)
        line.set_data(np.arange(n), densities)
        grid = updated_grid
        return [mat, line]

    ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()


if __name__ == '__main__':
    main()