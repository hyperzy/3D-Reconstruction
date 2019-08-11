import filesio
import display
import init
import evolution
import numpy as np
import cv2
import multiprocessing



'''
Note: 1. vector is represented as 1d array and used for 2d array when necessarily
      2. The dataset I use is a little bit special since the cameras surround the obejet
         and the origin of world coordinate system is ALMOST at the intersection of all
         the cameras optical shafts.
         If the dataset is not the case, try to do coordinate transformation first.
      3. I did not consider the case where there exist camera above or below the object(along z
         axis)
      4. I have not consider camera's projection on the surface is exactly at the border edge.
         It is also simple to implement, just take care when doing bilinear interpolation.
'''

all_params = []
origin_imgs = []
seg_imgs = []
resolution = init.resolution


if __name__ == "__main__":
    all_params = np.array(filesio.get_cam_param("viff.xml"))
    seg_imgs = filesio.ImageData("seg_images", 0).get_data()
    for it in seg_imgs:
        it = cv2.threshold(it, 128, 255, cv2.THRESH_BINARY)
    gray_imgs = filesio.ImageData("images", 0).get_data()
    '''
    since we test with 8 images, here we only retrieve 8 cameras
    '''
    all_params = all_params[[0, 4, 9, 13, 18, 22, 27, 31]]
    # print(all_params[0].getIntrinsic())
    # display.show_img(seg_imgs)
    # display.show_3D(all_params)

    # k = all_params[0].getIntrinsic()
    # k_inv = np.linalg.inv(k)
    # sil_coord = np.where(seg_imgs[0] != 0)
    # print(np.sort(sil_coord[0]))
    # index = np.argmin(sil_coord[0])
    # v = sil_coord[0][index]
    # u = sil_coord[1][index]
    # I_homo = np.array([u, v, 1]).reshape((3, 1))
    # z = 50
    # xc = z * k_inv @ I_homo
    # xc = np.append(xc, 1)
    # # print(xc)
    # rotation = all_params[0].getRotation()
    # motion = all_params[0].getMotion()
    # coord_transform_matrix = np.append(np.append(rotation.T, motion, axis=1), np.array([[0, 0, 0, 1]]), axis=0)
    # xw = coord_transform_matrix @ xc

    limits, point_set = init.determin_bound_coord([all_params[0], all_params[4], all_params[2], all_params[6]], [seg_imgs[0], seg_imgs[4], seg_imgs[2], seg_imgs[6]])

    print(limits)
    grid = init.init_grid()
    # interface, phi = init.init_level_set_function()
    # filesio.save_array("initial_grid_small.txt", phi, format="%d)
    phi = np.loadtxt("initial_grid_small.txt", dtype=np.int16).reshape(grid.shape[1:])
    interface = init.get_zero_set_index(phi)
    display.show_3D(all_params, testparam=point_set, testinterface=interface, testparam1=grid)

    object_evolution = evolution.Evolve(grid, phi, seg_imgs, gray_imgs, all_params, limits, resolution, bdbox_pointset=point_set)
    object_evolution.evolve()
    # object_visibility = visibility.Visibility(all_params[7], grid, phi, limits, resolution)
    # direction = object_visibility.determine_direction()
    # object_visibility.calculate_all()
    # nonvis = np.where(psi_list[0] < 0)
    # filesio.save_array("initial_grid.txt", phi)
    pass

    # radius = 5
    # arr = np.arange(-radius, radius + 1, 1) ** 2
    # mask2d = arr[:, None] + arr[None, :]
    # test2d = np.ones((100, 100), dtype=np.int16)
    # nx, ny = test2d.shape
    # test2d[int(nx/2) - radius: int(nx/2) + radius + 1, int(ny/2) - radius: int(ny/2) + radius + 1] = mask2d
    # zero_ls_index = np.where((test2d < (radius + 0.5) ** 2) & (test2d >= (radius - 0.5)** 2))
    # object_fmm = fmm2d.Fmm((nx, ny), zero_ls_index)
    # output = object_fmm.get_grid()
    # pass
