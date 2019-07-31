import numpy as np
from enum import enum
import math


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
        self.inf = 1 << 15
        self.grid[source_set_index] = 0
        # status 0 as accepted,
        # status 1 as tentative
        # status 2 as faraway
        self.status_grid = 2 * np.ones(shape, dtype=np.uint8)
        self.status_grid[source_set_index] = Status.tentative
        # 2D case #############################
        self.__neighbor_mask = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    # try to find tentative neighbors, k is defaulty set to None
    # since I need to test 2D first.
    # return the index of neighbors
    def find_tentative_neighbors(self, i, j, k=None):
        potential_neighbor = self.__neighbor_mask + np.array([i, j])
        potential_neighbor = filter(self.check_boundary, potential_neighbor)
        potential_neighbor = filter(lambda x: self.status_grid[x[0], x[1]] != Status.faraway, potential_neighbor)
        ####  3d case need to be changed here
        return (potential_neighbor[:, 0], potential_neighbor[:, 1])


    def check_boundary(self, point):
        return 0 <= point[0] < self.grid.shape[0]\
                & 0 <= point[1] < self.grid.shape[1]\
                # & 0 <= point[2] <= self.grid.shape[2]


    # try to find computable neighbor points
    def find_computable_neighbors(self, i, j, k=None):
        ####  3d case need to be changed here
        neighbors = self.__neighbor_mask + np.array([i, j])
        potential_neighbor = filter(self.check_boundary, potential_neighbor)
        return (neighbor[:, 0], neighbor[:, 1])


    # return index of tentative grid
    def get_tentative_index(self):
        return np.where(self.status_grid == Status.tentative)


    # march the grid by approximately solving Eikonal Equation
    def marching(self):
        speed = 1
        # minval = self.grid.min()
        # # create temperate minimum value to determine if 
        # # any larger value decreased
        # temp_minval = minval    
        # keep marching before all points get updated
        while(np.all(self.status_grid != Status.faraway)):
            minval = self.grid[self.get_tentative_index].min()
            min_val_index = np.where(self.grid == minval)
            # look tentative grid and update status
            for i, j in zip(min_val_index[0], min_val_index[1]):
                ####  3d case need to be changed here
                self.status_grid[i, j] = Status.accepted
                tentative_neighbor = self.find_tentative_neighbors(i, j)
                self.status_grid[tentative_neighbor] = Status.tentative
                # then use the algorithm introduced in https://blog.csdn.net/lusongno1/article/details/88409735 to approximate Eikonal Equation
                ############# to check: if i need to exclude tentative grid to ensure only once update
                for i_tentative, j_tentative in zip(tentative_neighbor[0], tentative_neighbor[1]):
                    computable_neighbor = self.find_computable_neighbors(i_tentative, j_tentative)
                    val_i_min = self.inf
                    val_j_min = self.inf
                    for i_computable, j_computable in zip(computable_neighbor[0], computable_neighbor[1]):
                        if j_computable == j_tentative:
                            val_i_min = min(val_i_min, self.grid[i_computable, j_computable])
                        elif i_computable == i_tentative:
                            val_j_min = min(val_j_min, self.grid[i_computable, j_computable])
                    if abs(val_j_min - val_i_min) < speed:
                        t = round((val_i_min + val_j_min + math.sqrt(1 / speed - pow(val_i_min - val_j_min, 2))) / 2)
                    else:
                        t = min(val_i_min, val_j_min) + 1 / speed

                    self.grid[i_tentative, j_tentative] = min(t, self.grid[i_tentative, j_tentative])


    def get_grid(self):
        self.marching()
        return self.grid

