import numpy as np
import functools
from itertools import combinations
import multiprocessing
import visibility
import display


psi_list = []


# callback function
def callback(args):
    psi_list[args[0]] = args[1]


# return divergence of 3D array
def divergence(x):
    return functools.reduce(np.add, np.gradient(x))


class Evolve(object):
    def __init__(self, grid, phi, seg_imgs, gray_imgs, cam_params, limits, resolution, bdbox_pointset=None):
        self.grid = grid
        self.psi_list = np.array
        self.phi = phi
        self.seg_imgs = np.array(seg_imgs)
        self.gray_imgs = np.array(gray_imgs)
        self.cam_params = np.array(cam_params)
        self.limits = limits
        self.resolution = resolution
        self.factor = np.array([0.3, 0.3, 0.4])
        self.epsilon = 0.01
        self.dirac = np.zeros(self.phi.shape, dtype=np.float64)
        self.Phi = np.ones(self.phi.shape, dtype=np.uint32)
        self.Phi.fill(1 << 15)
        # store projection coordinates regarding each cameras, 2D
        self.projection = []
        # to store which points are out of image boundary regarding each camera, 2D
        self.valid_visible = []
        self.img_boundary = seg_imgs[0].shape
        # convert grid to 2d array to accelerate iterating efficiency
        self.grid_2d = np.column_stack((self.grid.transpose((1, 2, 3, 0)).reshape((-1, 3)).copy(),
                                        np.array([1] * (self.phi.shape[0] * self.phi.shape[1] * self.phi.shape[2]))))
        self.accumulator = 0
        self.projection_ravel_index = 0
        self.region_energy = np.zeros(self.phi.shape, dtype=np.int32)
        self.region_energy.fill(1 << 15)
        self.flag_initial = True
        self.bdbox_pointset = bdbox_pointset
        # self.temp_point3d_coord = np.array
        for i in range(len(cam_params)):
            # global psi_list
            psi_list.append(np.array([]))

    def update_psi_list(self, psi_list):
        p = multiprocessing.Pool()
        for i in range(self.cam_params.shape[0]):
            p.apply_async(func=visibility.multiprocess_handler, args=(i, self.cam_params[i], self.grid, self.phi, self.limits, self.resolution),
                          callback=callback)
        p.close()
        p.join()
        self.psi_list = np.array([(i >= 0).reshape((-1, 1)) for i in psi_list])

    def check_img_boundary(self, j, i):
        return 0 <= i < self.img_boundary[0]\
                and 0 <= j < self.img_boundary[1]

    def construct_drac(self):
        np.add(self.epsilon**2, self.phi**2, out=self.dirac)
        np.divide(1, self.dirac, out=self.dirac)
        factor = self.epsilon / np.pi
        np.multiply(self.dirac, factor, out=self.dirac)

    # compute Phi and region-consistency energy
    def compute_Phi__and_RE(self):
        # for i in self.psi_list.shape[0]:
        #     for j in self.psi_list.shape[1]:
        #         for k in self.psi_list.shape[2]:
        #             # finding which camera(s) can see grid point (i, j, k)
        #             visible_index = np.where(self.psi_list[:, i, j, k] & self.valid_visible)
        #             if visible_index:
        #                 # find combinations of visible camera
        #                 comb = [c for c in combinations(visible_index[0], 2)]


        Phi_ravel = self.Phi.ravel()
        for i in range(self.phi.ravel().shape[0]):
            ############ here attention is needed
            visible_index = np.where(self.psi_list[:, i, 0] & self.valid_visible[:, i, 0])
            if visible_index:
                # find conbinations of visible cameras
                comb = [c for c in combinations(visible_index[0], 2)]
                self.projection_ravel_index = i
                map(self.compute_se, comb)
                Phi_ravel[i] = self.accumulator
                self.accumulator = 0

            if not self.flag_initial:
                pass
            else:
                visible_index = np.where(self.valid_visible[:, i, 0])
                region_energi_ravel = self.region_energy.ravel()
                if visible_index:
                    num_view = len(visible_index[0])
                    min_gray_value = np.min(self.gray_imgs[(visible_index[0],
                                                            self.projection[(visible_index[0], np.array([i] * num_view))][1],
                                                            self.projection[(visible_index[0], np.array([i] * num_view))][0])])
                    region_energi_ravel[i] = (255 - min_gray_value) ** 2 - min_gray_value ** 2

    # compute square error between two images
    def compute_se(self, pair):
        self.accumulator += (self.gray_imgs[pair[0]][self.projection[pair[0], self.projection_ravel_index, -2::-1]]
                             - self.gray_imgs[pair[1]][self.projection[pair[1], self.projection_ravel_index, -2::-1]]) ** 2

    def aux_filter(self, cam_index):
        return self.check_img_boundary(self.projection[cam_index])

    def compute_valid_projection(self):
        for i in self.cam_params:
            projection = self.grid_2d @ i.P.T
            np.divide(projection, projection[:, 2].reshape((-1, 1)), out=projection)
            valid_visible = np.where((projection[:, 0] >= 0) & (projection[:, 0] < self.img_boundary[1])\
                                     & (projection[:, 1] >= 0) & (projection[:, 1] < self.img_boundary[0]))
            self.valid_visible.append(valid_visible)
            self.projection.append(np.round(projection))
        self.projection = np.array(self.projection)
        self.valid_visible = np.array(self.valid_visible)



    def evolve(self):
        self.construct_drac()
        self.compute_valid_projection()
        timestep = 0.02
        interation_count = 0
        new_phi = self.phi.copy()
        all_term_result = np.zeros(self.phi.shape, dtype=np.float64)
        while interation_count < 500:
            self.update_psi_list(psi_list)
            # display.show_3D(all_params, testparam=point_set, testinterface=interface, testparam1=grid,
            #                 nonvisible=nonvis)
            # display.show_3D(self.cam_params, testparam=self.bdbox_pointset, testinterface=find_interface(self.phi),
            #                 testparam1=self.grid)
            self.compute_Phi__and_RE()

            # term 1
            phi_gradient = np.gradient(self.phi)
            div_phi = divergence(phi_gradient)
            phi_over_abs_phi = np.divide(phi_gradient, np.abs(phi_gradient))
            div_phi_over_abs_phi = divergence(phi_over_abs_phi)
            np.add(all_term_result, np.subtract(div_phi - div_phi_over_abs_phi), out=all_term_result)
            np.multiply(all_term_result, self.factor[0], out=all_term_result)

            # term 2
            Phi_gradient = np.gradient(self.Phi)
            temp_term = np.multiply(Phi_gradient, phi_over_abs_phi)
            np.add(temp_term, np.multiply(Phi_gradient, div_phi_over_abs_phi), out=temp_term)
            np.multiply(self.dirac, temp_term, out=temp_term)
            np.multiply(self.factor[1], temp_term, out=temp_term)
            np.add(all_term_result, temp_term, out=all_term_result)

            # term 3
            temp_term = np.multiply(self.dirac, self.region_energy)
            np.multiply(self.factor[2], temp_term)
            np.add(all_term_result, temp_term, out=all_term_result)

            np.add(self.phi, np.multiply(timestep, all_term_result), out=new_phi)

            self.phi, new_phi = new_phi, self.phi


def find_interface(phi):
    return np.where((phi > -0.5) & (phi < 0.5))