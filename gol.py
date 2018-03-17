import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from grid import get_density, grid_to_number, is_periodic, update_grid, init_grid
import bisect


INITIAL_DENSITY = 0.4

HEIGHT = 20
WIDTH = 20

logging = "{:>20}{:>20}{:>20}"


def main():
    fig, ax = plt.subplots(1, 2)
    grid = init_grid(HEIGHT, WIDTH, INITIAL_DENSITY)
    density = get_density(grid)
    mat = ax[0].matshow(grid, cmap=plt.cm.gray_r)
    densities = [density]
    state_ids = [grid_to_number(grid)]
    line, = ax[1].plot([0], densities)
    detected_periodicity = is_periodic(state_ids)
    print(logging.format(0, density, detected_periodicity))

    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0, 50)

    # generator of frames that stops when a periodicity is detected
    def gen():
        nonlocal detected_periodicity
        i = 0
        while not detected_periodicity:
            i += 1
            yield i

    def update(i):
        nonlocal grid, densities, state_ids, detected_periodicity
        updated_grid = update_grid(grid)
        updated_density = get_density(updated_grid)
        updated_state_id = grid_to_number(updated_grid)
        densities.append(updated_density)
        # we insert the state ids sorted in order to speed up the
        # periodicity detection
        bisect.insort(state_ids, updated_state_id)
        n = len(densities)
        xmin, xmax = ax[1].get_xlim()
        if n >= xmax:
            ax[1].set_xlim(xmin, xmax*2)
            ax[1].figure.canvas.draw()
        mat.set_data(updated_grid)
        line.set_data(np.arange(n), densities)
        grid = updated_grid

        detected_periodicity = is_periodic(state_ids)
        print(logging.format(i, updated_density, detected_periodicity))

        return [mat, line]

    ani = animation.FuncAnimation(fig, update, frames=gen, interval=50, blit=True)
    plt.show()


if __name__ == '__main__':
    main()