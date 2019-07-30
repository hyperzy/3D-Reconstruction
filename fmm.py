import numpy as np
from enum import enum


class Status(Enum):
    accepted = 1
    tentative = 2
    faraway = 3


class Fmm(object):
    def __init__(self, shape, source_set_index):
        self.__dfault_type = np.int16
        self.__dfault_inf_shift = 15
        self.grid = np.ones(shape, dtype=self.__dfault_type)
        np.left_shift(self.grid, 15, out=self.grid)
        self.grid(source_set_index) = 0
        # status 0 as accepted,
        # status 1 as tentative
        # status 2 as faraway
        self.status_grid = 2 * np.ones(shape, dtyp=np.uint8)
        self.status_grid(source_set_index) = Status.accepted
        # 2D case #############################
        self.__neighbor_mask = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    # try to find tentative neighbors, k is defaulty set to None
    # since I need to test 2D first
    def find_tentative_neighbors(self, i, j, k=None):
        potential_neighbor = self.__neighbor_mask + np.array([i, j])
        potential_neighbor = filter(check_boundary, potential_neighbor)
        potential_neighbor = filter(lambda x: self.status_grid)


    def check_boundary(self, point):
        return 0 <= point[0] < self.grid.shape[0]\
                & 0 <= point[1] < self.grid.shape[1]\
                # & 0 <= point[2] <= self.grid.shape[2]