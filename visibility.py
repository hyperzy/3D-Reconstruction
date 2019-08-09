import numpy as np
from enum import Enum
import filesio
import math
import multiprocessing
import main

np.seterr(all="raise")


class Direction(Enum):
    # sweeping from x negative to x positive
    yz_pos = 1
    # sweeping from x positive to x negative
    yz_neg = 2
    xz_pos = 3
    xz_neg = 4
    xy_pos = 5
    xy_neg = 6


class Visibility(object):
    def __init__(self, cam_param, grid, phi, limits, resolution):
        self.cam_param = cam_param
        self.camera_coord = self.cam_param.getMotion()
        # self.camera_coord = np.array([54, 0.6, 10.3])
        # self.camera_coord[2] = 10.3
        self.grid = grid
        self.phi = phi
        self.limits = limits
        self.direction = self.determine_direction()
        self.resolution = resolution
        self.psi = np.zeros(phi.shape, dtype=np.float32)
        # 99% accurate
        self.err = 0.01
        self.tolerance = self.resolution * self.err
        self.float_accuracy = 1e-6

    # determine sweeping plane and direction
    def determine_direction(self):
        # (x_neg_limits, y_neg_limits, z_neg_limits)
        pointA = np.array([self.limits[0], self.limits[2], self.limits[4]])
        # (x_pos_limits, y_neg_limits, z_neg_limits)
        pointB = np.array([self.limits[1], self.limits[2], self.limits[4]])
        pointC = np.array([self.limits[1], self.limits[3], self.limits[4]])
        pointD = np.array([self.limits[0], self.limits[3], self.limits[4]])
        # (x_neg_limits, y_neg_limits, z_pos_limits)
        pointE = np.array([self.limits[0], self.limits[2], self.limits[5]])
        pointF = np.array([self.limits[1], self.limits[2], self.limits[5]])
        pointG = np.array([self.limits[1], self.limits[3], self.limits[5]])
        pointG = np.array([self.limits[0], self.limits[3], self.limits[5]])

        normalA = np.array([1, -1, 0])
        normalB = np.array([0, -1, 0])
        normalC = np.array([-1, -1, 0])
        normalD = np.array([1, 0, 0])

        # for now, since we know the there is no camera on the top of bottom of 
        # the object, we only consider four cases below.
        if self.determine_position(self.camera_coord, pointA, normalA)\
            and self.determine_position(self.camera_coord, pointA, normalB)\
            and self.determine_position(self.camera_coord, pointB, normalC):
            return Direction.xz_pos
        elif self.determine_position(self.camera_coord, pointB, -normalC)\
            and self.determine_position(self.camera_coord, pointB, normalD)\
            and self.determine_position(self.camera_coord, pointC, normalA):
            return Direction.yz_neg
        elif self.determine_position(self.camera_coord, pointC, -normalA)\
            and self.determine_position(self.camera_coord, pointC, -normalB)\
            and self.determine_position(self.camera_coord, pointD, -normalC):
            return Direction.xz_neg
        elif self.determine_position(self.camera_coord, pointD, normalC)\
            and self.determine_position(self.camera_coord, pointD, -normalD)\
            and self.determine_position(self.camera_coord, pointA, -normalA):
            return Direction.yz_pos
        else:
            print("Error Camera Position. Try to do coordinate system transformation first")
            exit()

    # determine the relative position of points and plane
    # return true if the points lies on normal direction side
    def determine_position(self, point_coord, plane_point_coord, plane_normal):
        return (point_coord - plane_point_coord).dot(plane_normal) >= 0

    # determine if the camera does lie on one of the grid
    def is_on_grid(self):
        # also, only consider two cases
        if self.direction == Direction.xz_pos or self.direction == Direction.xz_neg:
            x_index = (self.camera_coord[0] - self.limits[0]) / self.resolution
            z_index = (self.camera_coord[2] - self.limits[2]) / self.resolution
            if x_index == math.floor(x_index) and z_index == math.floor(z_index):
                return True
            else:
                return False
        elif self.direction == Direction.yz_pos or self.direction == Direction.yz_neg:
            y_index = (self.camera_coord[1] - self.limits[1]) / self.resolution
            z_index = (self.camera_coord[2] - self.limits[2]) / self.resolution
            if y_index == math.floor(y_index) and z_index == math.floor(z_index):
                return True
            else:
                return False

    # calculate visibility
    # algorightm is refered from "Visibility and Dynamics in a PDE Based Implicity Framework"
    def calculate_all(self):
        psi = self.psi
        x_index_float = (self.camera_coord[0] - self.limits[0]) / self.resolution
        y_index_float = (self.camera_coord[1] - self.limits[2]) / self.resolution
        z_index_float = (self.camera_coord[2] - self.limits[4]) / self.resolution
        if math.fabs(x_index_float - round(x_index_float)) < self.float_accuracy:
            x_index_float = float(round(x_index_float))
        if math.fabs(y_index_float - round(y_index_float)) < self.float_accuracy:
            y_index_float = float(round(y_index_float))
        if math.fabs(z_index_float - round(z_index_float)) < self.float_accuracy:
            z_index_float = float(round(z_index_float))

        if self.direction == Direction.xz_pos or self.direction == Direction.xz_neg:
            flag = self.direction == Direction.xz_pos
            x_limit, y_limit, z_limit = self.phi.shape
            # first initiatize the visibility value of four closest surfaces
            # around the camera's projection on the cube surface
            x_smaller_index = math.floor(x_index_float)
            x_smaller_index = x_limit - 2 if x_smaller_index >= x_limit else (0 if x_smaller_index < 0 else x_smaller_index)
            x_larger_index = x_smaller_index + 1
            z_smaller_index = math.floor(z_index_float)
            z_smaller_index = z_limit - 2 if z_smaller_index >= z_limit else (0 if z_smaller_index < 0 else z_smaller_index)
            z_larger_index = z_smaller_index + 1
            self.psi[:, 0 if self.direction == Direction.xz_pos else y_limit - 1, :] = self.phi[:, 0 if self.direction == Direction.xz_pos else y_limit - 1, :]
            x_iter_arr = np.array([x_smaller_index, x_larger_index])
            z_iter_arr = np.array([z_smaller_index, z_larger_index])

            # determine if camera's coordinates excceds the bounding box
            x_upper_overflow, x_lower_overflow, z_upper_overflow, z_lower_overflow = False, False, False, False
            if x_index_float > x_limit - 1:
                x_upper_overflow = True
                self.psi[x_limit - 1, :, :] = self.phi[x_limit - 1, :, :]
                x_iter_arr = np.array([x_smaller_index])
            elif x_index_float < 0:
                x_lower_overflow = True
                self.psi[0, :, :] = self.phi[0, :, :]
                x_iter_arr = np.array([x_larger_index])
            if z_index_float > z_limit - 1:
                z_upper_overflow = True
                self.psi[:, :, z_limit - 1] = self.phi[:, :, z_limit - 1]
                z_iter_arr = np.array([z_smaller_index])
            elif z_index_float < 0:
                self.psi[:, :, 0] = self.phi[:, :, 0]
                z_lower_overflow = True
                z_iter_arr = np.array([z_larger_index])
            # Since I use floor() + 1 instead of ceil(), camera's projectoin on grid case is generalized as not_on_grid() case
            for y_index in np.arange(1 if flag else y_limit - 2,
                                     y_limit if flag else -1,
                                     1 if flag else -1):
                for z_index in z_iter_arr:
                    for x_index in np.arange(1 if x_lower_overflow else 0, x_limit - 1 if x_upper_overflow else x_limit):
                        self.psi[x_index, y_index, z_index] = self.calculate_point(x_index, y_index, z_index)
                for x_index in x_iter_arr:
                    for z_index in np.append(np.arange(0, z_smaller_index), np.arange(z_larger_index + 1, z_limit)):
                        self.psi[x_index, y_index, z_index] = self.calculate_point(x_index, y_index, z_index)

            print("finished initial surface")
            # then compute visibility of four quadrants
            quadrant_control = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
            for control in quadrant_control:
                for y_index in np.arange(1 if flag else y_limit - 2,
                                         y_limit if flag else -1,
                                         1 if flag else -1):
                    # print("processing plane Y = ", y_index)
                    for z_index in np.arange(control[1] + (z_larger_index if control[1] > 0 else z_smaller_index),
                                                 z_limit if control[1] > 0 else -1,
                                                 control[1]):
                        for x_index in np.arange(control[0] + (x_larger_index if control[0] > 0 else x_smaller_index),
                                                 x_limit if control[0] > 0 else -1,
                                                 control[0]):
                            self.psi[x_index, y_index, z_index] = self.calculate_point(x_index, y_index, z_index)

                print("finished one quadrant")

        # still, only consider two cases
        if self.direction == Direction.yz_pos or self.direction == Direction.yz_neg:
            flag = self.direction == Direction.yz_pos
            x_limit, y_limit, z_limit = self.phi.shape
            # first initiatize the visibility value of four closest points
            # around the camera's projection on the cube surface
            y_smaller_index = math.floor(y_index_float)
            y_smaller_index = y_limit - 2 if y_smaller_index >= y_limit else (0 if y_smaller_index < 0 else y_smaller_index)
            y_larger_index = y_smaller_index + 1
            z_smaller_index = math.floor(z_index_float)
            z_smaller_index = z_limit - 2 if z_smaller_index >= z_limit else (0 if z_smaller_index < 0 else z_smaller_index)
            z_larger_index = z_smaller_index + 1
            self.psi[0 if self.direction == Direction.yz_pos else x_limit - 1, :, :] = self.phi[0 if self.direction == Direction.yz_pos else x_limit - 1, :, :]
            y_iter_arr = np.array([y_smaller_index, y_larger_index])
            z_iter_arr = np.array([z_smaller_index, z_larger_index])

            # dtermine if camera's coordinates excceds the bounding box
            y_upper_overflow, y_lower_overflow, z_upper_overflow, z_lower_overflow = False, False, False, False
            if y_index_float > y_limit - 1:
                y_upper_overflow = True
                self.psi[:, y_limit - 1, :] = self.phi[:, y_limit - 1, :]
                y_iter_arr = np.array([y_smaller_index])
            elif y_index_float < 0:
                x_lower_overflow = True
                self.psi[:, 0, :] = self.phi[:, 0, :]
                y_iter_arr = np.array([y_larger_index])
            if z_index_float > z_limit - 1:
                z_upper_overflow = True
                self.psi[:, :, z_limit - 1] = self.phi[:, :, z_limit - 1]
                z_iter_arr = np.array([z_smaller_index])
            elif z_index_float < 0:
                self.psi[:, :, 0] = self.phi[:, :, 0]
                z_lower_overflow = True
                z_iter_arr = np.array([z_larger_index])
            # Since I use floor() + 1 instead of ceil(), on_grid() case is generalized as not_on_grid() case
            for x_index in np.arange(1 if flag else x_limit - 2,
                                     x_limit if flag else -1,
                                     1 if flag else -1):
                for z_index in z_iter_arr:
                    for y_index in np.arange(1 if y_lower_overflow else 0, y_limit - 1 if y_upper_overflow else y_limit):
                        self.psi[x_index, y_index, z_index] = self.calculate_point(x_index, y_index, z_index)
                for y_index in y_iter_arr:
                    for z_index in np.append(np.arange(0, z_smaller_index), np.arange(z_larger_index + 1, z_limit)):
                        self.psi[x_index, y_index, z_index] = self.calculate_point(x_index, y_index, z_index)

            print("finished initial surface")
            # then compute visibility of four quadrants
            quadrant_control = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
            for control in quadrant_control:
                for x_index in np.arange(1 if flag else x_limit - 2,
                                         x_limit if flag else -1,
                                         1 if flag else -1):
                    for z_index in np.arange(control[1] + (z_larger_index if control[1] > 0 else z_smaller_index),
                                                 z_limit if control[1] > 0 else -1,
                                                 control[1]):
                        for y_index in np.arange(control[0] + (y_larger_index if control[0] > 0 else y_smaller_index),
                                                 y_limit if control[0] > 0 else -1,
                                                 control[0]):
                            self.psi[x_index, y_index, z_index] = self.calculate_point(x_index, y_index, z_index)

                print("finished one quadrant")
                
    # calculate the visibility of given point
    def calculate_point(self, x_index, y_index, z_index):
        minval = self.phi[x_index, y_index, z_index]
        vec = self.camera_coord - self.grid[:, x_index, y_index, z_index]
        vec[np.where(np.fabs(vec) < self.float_accuracy)] = 0
        # choose a princible axis to increase
        argmax_axis = np.argmax(np.abs(vec))
        # direction variables
        x_dir = 1 if vec[0] > 0 else (0 if vec[0] == 0 else -1)
        y_dir = 1 if vec[1] > 0 else (0 if vec[1] == 0 else -1)
        z_dir = 1 if vec[2] > 0 else (0 if vec[2] == 0 else -1)

        # used to record the nearest base integer index
        cur_x_index = x_index
        cur_y_index = y_index
        cur_z_index = z_index

        # ray-tracing back
        if argmax_axis == 0:
            cur_x_coord, cur_y_coord, cur_z_coord = self.grid[:, x_index, y_index, z_index]
            next_x_coord, next_y_coord, next_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index + x_dir, y_index, z_index], Direction.yz_pos)
            # check error, if the error is less than tolerance, just assume the ray is along the grid edge
            # if math.fabs(next_y_coord - cur_y_coord) < self.tolerance:
            #     next_y_coord = cur_y_coord
            y_changed = cur_y_index + y_dir
            # if math.fabs(next_z_coord - cur_z_coord) < self.tolerance:
            #     next_z_coord = cur_z_coord
            z_changed = cur_z_index + z_dir

            # used to determine if the ray cross other planes instead of hiting X = x+x_dir*resolution
            y_determiner = self.grid[1, x_index, y_changed, z_index]
            z_determiner = self.grid[2, x_index, y_index, z_changed]
            next1_x_coord, next1_y_coord, next1_z_coord = next_x_coord, next_y_coord, next_z_coord
            next2_x_coord, next2_y_coord, next2_z_coord = next_x_coord, next_y_coord, next_z_coord
            # if ray crosses plane Y = y[y_changed], interpolation is computed on this plane.
            if (y_dir > 0 and next_y_coord > y_determiner) or (y_dir < 0 and next_y_coord < y_determiner):
                next1_x_coord, next1_y_coord, next1_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index, y_changed, z_index], Direction.xz_pos)
            # if ray crosses plane Z = z[z_changed], interpolation is computed on this plane.
            if (z_dir > 0 and next_z_coord > z_determiner) or (z_dir < 0 and next_z_coord < z_determiner):
                next2_x_coord, next2_y_coord, next2_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index, y_index, z_changed], Direction.xy_pos)
            # determine the neareast neighbor
            if x_dir:
                if next1_x_coord > next2_x_coord:
                    next_x_coord, next_y_coord, next_z_coord = next1_x_coord, next1_y_coord, next1_z_coord
                else:
                    next_x_coord, next_y_coord, next_z_coord = next2_x_coord, next2_y_coord, next2_z_coord
            else:
                if next1_x_coord < next2_x_coord:
                    next_x_coord, next_y_coord, next_z_coord = next1_x_coord, next1_y_coord, next1_z_coord
                else:
                    next_x_coord, next_y_coord, next_z_coord = next2_x_coord, next2_y_coord, next2_z_coord

            val = self.interpolation([next_x_coord, next_y_coord, next_z_coord], 
                [x_index + x_dir, cur_y_index, z_changed], [x_index + x_dir, y_changed, z_changed],
                [x_index + x_dir, cur_y_index, cur_z_index], [x_index + x_dir, y_changed, cur_z_index],
                Direction.yz_pos)
            minval = min(minval, val)

        elif argmax_axis == 1:
            cur_x_coord, cur_y_coord, cur_z_coord = self.grid[:, x_index, y_index, z_index]
            next_x_coord, next_y_coord, next_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index, y_index + y_dir, z_index], Direction.xz_pos)

            # check error, if the error is less than tolerance, just assume the ray is along the grid edge
            # if math.fabs(next_x_coord - cur_x_coord) < self.tolerance:
            #     next_x_coord = cur_x_coord
            x_changed = cur_x_index + x_dir
            # if math.fabs(next_z_coord - cur_z_coord) < self.tolerance:
            #     next_z_coord = cur_z_coord
            z_changed = cur_z_index + z_dir

            # used to determine if the ray cross other planes instead of hiting Y = y+y_dir*resolution
            x_determiner = self.grid[0, x_changed, y_index, z_index]
            z_determiner = self.grid[2, x_index, y_index, z_changed]
            next1_x_coord, next1_y_coord, next1_z_coord = next_x_coord, next_y_coord, next_z_coord
            next2_x_coord, next2_y_coord, next2_z_coord = next_x_coord, next_y_coord, next_z_coord
            # if ray crosses plane X = x[x_changed], interpolation is computed on this plane.
            if (x_dir > 0 and next_x_coord > x_determiner) or (x_dir < 0 and next_x_coord < x_determiner):
                next1_x_coord, next1_y_coord, next1_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_changed, y_index, z_index], Direction.yz_pos)

            # if ray crosses plane Z = z[z_changed], interpolation is computed on this plane.
            if (z_dir > 0 and next_z_coord > z_determiner) or (z_dir < 0 and next_z_coord < z_determiner):
                next2_x_coord, next2_y_coord, next2_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index, y_index, z_changed], Direction.xy_pos)
            # determine the neareast neighbor
            if y_dir:
                if next1_y_coord > next2_y_coord:
                    next_x_coord, next_y_coord, next_z_coord = next1_x_coord, next1_y_coord, next1_z_coord
                else:
                    next_x_coord, next_y_coord, next_z_coord = next2_x_coord, next2_y_coord, next2_z_coord
            else:
                if next1_y_coord < next2_y_coord:
                    next_x_coord, next_y_coord, next_z_coord = next1_x_coord, next1_y_coord, next1_z_coord
                else:
                    next_x_coord, next_y_coord, next_z_coord = next2_x_coord, next2_y_coord, next2_z_coord
                    
            # if ((x_dir and next_x_coord < x_determiner) or (-x_dir and next_x_coord > x_determiner))\
            #     and ((z_dir and next_z_coord < z_determiner) or (-z_dir and next_z_coord > z_determiner)):
            val = self.interpolation([next_x_coord, next_y_coord, next_z_coord],
                [cur_x_index, y_index + y_dir, z_changed], [x_changed, y_index + y_dir, z_changed],
                [cur_x_index, y_index + y_dir, cur_z_index], [x_changed, y_index + y_dir, cur_z_index],
                Direction.xz_pos)
            minval = min(minval, val)
                
        elif argmax_axis == 2:
            cur_x_coord, cur_y_coord, cur_z_coord = self.grid[:, x_index, y_index, z_index]
            next_x_coord, next_y_coord, next_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index, y_index, z_index], Direction.xy_pos)

            # check error, if the error is less than tolerance, just assume the ray is along the grid edge
            # if math.fabs(next_x_coord - cur_x_coord) < self.tolerance:
            #     next_x_coord = cur_x_coord
            x_changed = cur_x_index + x_dir
            # if math.fabs(next_y_coord - cur_y_coord) < self.tolerance:
            #     next_y_coord = cur_y_coord
            y_changed = cur_y_index + y_dir

            # used to determine if the ray cross other planes instead of hiting Y = y+y_dir*resolution
            x_determiner = self.grid[0, x_changed, y_index, z_index]
            y_determiner = self.grid[1, x_index, y_index, y_changed]
            next1_x_coord, next1_y_coord, next1_z_coord = next_x_coord, next_y_coord, next_z_coord
            next2_x_coord, next2_y_coord, next2_z_coord = next_x_coord, next_y_coord, next_z_coord
            # if ray crosses plane X = x[x_changed], interpolation is computed on this plane.
            if (x_dir > 0 and next_x_coord > x_determiner) or (x_dir < 0 and next_x_coord < x_determiner):
                next1_x_coord, next1_y_coord, next1_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_changed, y_index, z_index], Direction.yz_pos)

            # if ray crosses plane Y = y[y_changed], interpolation is computed on this plane.
            if (y_dir > 0 and next_y_coord > y_determiner) or (y_dir < 0 and next_y_coord < y_determiner):
                next1_x_coord, next1_y_coord, next1_z_coord = self.intersect(vec, self.grid[:, x_index, y_index, z_index], self.grid[:, x_index, y_changed, z_index], Direction.xz_pos)

            # determine the neareast neighbor
            if z_dir:
                if next1_z_coord > next2_z_coord:
                    next_x_coord, next_y_coord, next_z_coord = next1_x_coord, next1_y_coord, next1_z_coord
                else:
                    next_x_coord, next_y_coord, next_z_coord = next2_x_coord, next2_y_coord, next2_z_coord
            else:
                if next1_z_coord < next2_z_coord:
                    next_x_coord, next_y_coord, next_z_coord = next1_x_coord, next1_y_coord, next1_z_coord
                else:
                    next_x_coord, next_y_coord, next_z_coord = next2_x_coord, next2_y_coord, next2_z_coord 
                       
            # if ((x_dir and next_x_coord < x_determiner) or (-x_dir and next_x_coord > x_determiner))\
            #     and ((z_dir and next_z_coord < z_determiner) or (-z_dir and next_z_coord > z_determiner)):
            val = self.interpolation([next_x_coord, next_y_coord, next_z_coord],
                [cur_x_index, cur_y_index, z_index + z_dir], [x_changed, cur_y_index, z_index + z_dir],
                [cur_x_index, y_changed, z_index + z_dir], [x_changed, y_changed, z_index + z_dir],
                Direction.xy_pos)
            minval = min(val, minval)


        return minval

    # compute intersection between line and plane
    # @parameter direction is used to determine the normal of intersecting plane
    def intersect(self, vector, current_point_coord, plane_point_coord, direction):
        '''
        P = P_0 + vect{v} \cdot t,
        and t = \dfrac{vec{n} \cdot vec{v'}}{vec{n} \cdot vec{v}}
        '''
        # vector constructed by plane_point_coord - P_0
        vec_prime = plane_point_coord - current_point_coord
        if direction == Direction.xz_pos or direction == Direction.xz_neg:
            normal = np.array([0, 1, 0])
        elif direction == Direction.yz_pos or direction == Direction.yz_neg:
            normal = np.array([1, 0, 0])
        else:
            normal = np.array([0, 0, 1])

        return current_point_coord + vector * (normal.dot(vec_prime) / normal.dot(vector))

    # check tolerance
    # This method is used to approximate coordinate so as to decrease computation.
    # # this method remains to be implemented to imporve the efficiency by decrease
    # # the computation of interpolation
    def check_err(self, cur_x_coord, cur_y_coord, cur_z_coord, \
                  next_x_coord, next_y_coord, next_z_coord, \
                  direction):
        pass

    # Bilinear Interpolation
    def interpolation(self, p_coord, p0_index, p1_index, p2_index, p3_index, direction):
        p0_x_index, p0_y_index, p0_z_index = p0_index
        p1_x_index, p1_y_index, p1_z_index = p1_index
        p2_x_index, p2_y_index, p2_z_index = p2_index
        p3_x_index, p3_y_index, p3_z_index = p3_index
        p0_coord = self.grid[:, p0_x_index, p0_y_index, p0_z_index]
        p3_coord = self.grid[:, p3_x_index, p3_y_index, p3_z_index]

        p0 = self.psi[p0_x_index, p0_y_index, p0_z_index]
        p1 = self.psi[p1_x_index, p1_y_index, p1_z_index]
        p2 = self.psi[p2_x_index, p2_y_index, p2_z_index]
        p3 = self.psi[p3_x_index, p3_y_index, p3_z_index]
        # for point whose neighbors all have positive or negative value,
        # sign of the value matters.
        # if (p0 > 0 and p1 > 0 and p2 > 0 and p3 > 0)\
        #         or (p0 < 0 and p1 < 0 and p2 < 0 and p3 < 0):
        #     val = (p0 + p1 + p2 + p3) / 4
        #     return val
        if direction == Direction.xz_neg or direction == Direction.xz_pos:
            denominator = (p3_coord[2] - p0_coord[2]) * (p3_coord[0] - p0_coord[0])
            val = ((p3_coord[2] - p_coord[2]) *(p0 * (p3_coord[0] - p_coord[0]) +
                    p1 * (p_coord[0] - p0_coord[0])) + 
                    (p_coord[2] - p0_coord[2]) * (p2 * (p3_coord[0] - p_coord[0]) + 
                    p3 * (p_coord[0] - p0_coord[0]))) / denominator
        elif direction == Direction.yz_neg or direction == Direction.yz_pos:
            denominator = (p3_coord[2] - p0_coord[2]) * (p3_coord[1] - p0_coord[1])
            val = ((p3_coord[2] - p_coord[2]) *(p0 * (p3_coord[1] - p_coord[1]) + 
                    p1 * (p_coord[1] - p0_coord[1])) + 
                    (p_coord[2] - p0_coord[2]) * (p2 * (p3_coord[1] - p_coord[1]) + 
                    p3 * (p_coord[1] - p0_coord[1]))) / denominator
        else:
        # xy plane
            denominator = (p3_coord[1] - p0_coord[1]) * (p3_coord[0] - p0_coord[0])
            val = ((p3_coord[1] - p_coord[1]) *(p0 * (p3_coord[0] - p_coord[0]) + 
                    p1 * (p_coord[0] - p0_coord[0])) + 
                    (p_coord[1] - p0_coord[1]) * (p2 * (p3_coord[0] - p_coord[0]) + 
                    p3 * (p_coord[0] - p0_coord[0]))) / denominator
        return val


# multi-processing handler
def multiprocess_handler(id, *args):
    print("%s is working" % multiprocessing.current_process().name)
    object_vis = Visibility(*args)
    object_vis.calculate_all()
    return id, object_vis.psi
