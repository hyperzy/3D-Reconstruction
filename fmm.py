import numpy as np
from enum import Enum
import math
import decimal


class Status(Enum):
    accepted = 1
    tentative = 2
    faraway = 3


class Fmm(object):
    def __init__(self, shape, source_set_index):
        self.__default_type = np.int16
        self.__default_int_shift = 15
        self.grid = (1 << self.__default_int_shift) * np.ones(shape, dtype=self.__default_type)
        self.inf = 1 << self.__default_int_shift
        self.grid[source_set_index] = 0
        # status 0 as accepted,
        # status 1 as tentative
        # status 2 as faraway
        self.status_grid = Status.faraway.value * np.ones(shape, dtype=np.uint8)
        self.status_grid[source_set_index] = Status.tentative.value
        self.__neighbor_mask = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])

    # try to find tentative neighbors, k is defaultlyset to None
    # since I need to test 2D first.
    # return the index of neighbors
    def find_tentative_neighbors(self, i, j, k):
        potential_neighbor = self.__neighbor_mask + np.array([i, j, k])
        potential_neighbor = np.array(list(filter(self.check_boundary, potential_neighbor)))
        potential_neighbor = np.array(list(filter(lambda x: self.status_grid[x[0], x[1], x[2]] == Status.faraway.value, potential_neighbor)))
        ####  3d case need to be changed here
        if len(potential_neighbor) == 0:
            return None
        else:
            return (potential_neighbor[:, 0], potential_neighbor[:, 1], potential_neighbor[:, 2])

    def check_boundary(self, point):
        return (0 <= point[0] < self.grid.shape[0])\
                & (0 <= point[1] < self.grid.shape[1])\
                & (0 <= point[2] < self.grid.shape[2])

    # try to find computable neighbor points
    def find_computable_neighbors(self, i, j, k):
        neighbor = self.__neighbor_mask + np.array([i, j, k])
        neighbor = np.array(list(filter(self.check_boundary, neighbor)))
        return (neighbor[:, 0], neighbor[:, 1], neighbor[:, 2])

    # return index of tentative grid
    def get_tentative_index(self):
        return np.where(self.status_grid == Status.tentative.value)

    # march the grid by approximately solving Eikonal Equation
    def marching(self):
        speed = 1
        # minval = self.grid.min()
        # # create temperate minimum value to determine if 
        # # any larger value decreased
        # temp_minval = minval    
        # keep marching before all points get updated
        while(np.any(self.status_grid == Status.faraway.value)):
            tentative_index = self.get_tentative_index()
            if len(tentative_index[0]) == 0:
                continue
            minval = self.grid[tentative_index].min()
            print("Processing value: ", minval)
            min_val_index = np.where(self.grid == minval)
            # look for tentative grid and update status
            for i, j, k in zip(min_val_index[0], min_val_index[1], min_val_index[2]):
                ####  3d case need to be changed here
                self.status_grid[i, j, k] = Status.accepted.value
                tentative_neighbor = self.find_tentative_neighbors(i, j, k)
                if tentative_neighbor == None:
                    continue
                self.status_grid[tentative_neighbor] = Status.tentative.value
                # then use the algorithm introduced in https://blog.csdn.net/lusongno1/article/details/88409735 to approximate Eikonal Equation
                ############# to check: if i need to exclude tentative grid to ensure only once update
                for i_tentative, j_tentative, k_tentative in zip(tentative_neighbor[0], tentative_neighbor[1], tentative_neighbor[2]):
                    computable_neighbor = self.find_computable_neighbors(i_tentative, j_tentative, k_tentative)
                    val_i_min = self.inf
                    val_j_min = self.inf
                    val_k_min = self.inf
                    for i_computable, j_computable, k_computable in zip(computable_neighbor[0], computable_neighbor[1],computable_neighbor[2]):
                        if (j_computable == j_tentative) & (i_computable == i_tentative):
                            val_k_min = min(val_k_min, self.grid[i_computable, j_computable, k_computable])
                        elif (i_computable == i_tentative) & (k_computable == k_tentative):
                            val_j_min = min(val_j_min, self.grid[i_computable, j_computable, k_computable])
                        elif (j_computable == j_tentative) & (k_computable == k_tentative):
                            val_i_min = min(val_i_min, self.grid[i_computable, j_computable, k_computable])
                    # create a list for sorting these three values and sort them descently
                    val_list = [val_i_min, val_j_min, val_k_min]
                    val_list.sort(reverse=True)
                    # a1 >= a2 >= a3
                    a1, a2, a3 = val_list

                    if (a1 - a2)**2 + (a1 - a3)**2 < 1 / (speed**2):
                        t = round((a1 + a2 + a3 + math.sqrt(3 / (speed**2) - (a1 - a2)**2 - (a1 - a3)**2 - (a2 - a3)**2)) / 3 + 0.5)
                    elif ((a1 - a2)**2 + (a1 - a3)**2 >= 1 / (speed**2)) & (abs(a2 - a3) < 1 / speed):
                        t = round((a2 + a3 + math.sqrt(2 / (speed ** 2) - (a2 - a3)**2)) / 2 + 0.5)
                    else:
                        t = a3 + 1 / speed

                    self.grid[i_tentative, j_tentative, k_tentative] = min(t, self.grid[i_tentative, j_tentative, k_tentative])

    def get_grid(self):
        self.marching()
        return self.grid

