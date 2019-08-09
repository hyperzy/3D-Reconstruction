import fmm
import numpy as np
import cv2
import time

resolution = 0.2
limits = []
bound_coord = []
dt = np.float32


# z is the depth where the 3D point locates at
def compute_back_rays(cam_params, seg_img, z=50, direction="top"):
    k = cam_params.getIntrinsic()
    k_inv = np.linalg.inv(k)
    rotation = cam_params.getRotation()
    motion = cam_params.getMotion()
    sil_coord = np.where(seg_img != 0)
    if direction == "top":
        index = np.where(sil_coord[0] == sil_coord[0].min())
    elif direction == "bottom":
        index = np.where(sil_coord[0] == sil_coord[0].max())
    elif direction == "left":
        index = np.where(sil_coord[1] == sil_coord[1].min())
    elif direction == "right":
        index = np.where(sil_coord[1] == sil_coord[1].max())
    else:
        print("error direction")
        exit()

    point3d_set = []
    for i in [0, -1]:
        v = sil_coord[0][index[0][i]]
        u = sil_coord[1][index[0][i]]
        I_homo = np.array([u, v, 1]).reshape((3, 1))
        xc = z * k_inv @ I_homo
        xc = np.append(xc, 1)
        # print(xc)
        coord_transform_matrix = np.append(np.append(rotation.T, motion.reshape(3, 1), axis=1), np.array([[0, 0, 0, 1]]), axis=0)
        xw = coord_transform_matrix @ xc
        xw = np.resize(xw, 3)
        point3d_set.append(xw)

    return point3d_set


# return one point coordinate on the intersection line
# the parameters is an array of two sets of two points
def compute_intersection(cam_params_arr, point3d_set_arr):
    c1 = cam_params_arr[0].getMotion()
    c2 = cam_params_arr[1].getMotion()
    norm1 = np.cross(point3d_set_arr[0][0] - c1, point3d_set_arr[0][1] - c1)
    norm2 = np.cross(point3d_set_arr[1][0] - c2, point3d_set_arr[1][1] - c2)

    norm1 = np.divide(norm1, np.linalg.norm(norm1), out=norm1)
    norm2 = np.divide(norm2, np.linalg.norm(norm2), out=norm2)
    # intersection line vector
    vecline_inter = np.cross(norm1, norm2)
    vecline_inter = np.divide(vecline_inter, np.linalg.norm(vecline_inter), out=vecline_inter)

    # aux line vector
    veclin_aux = np.cross(vecline_inter, norm1)
    veclin_aux = np.divide(veclin_aux, np.linalg.norm(veclin_aux), out=veclin_aux)

    # the result come from the algebra
    # which is vec(c2 - c1)\cdot norm2 = d*veclin_aux\cdot norm2
    aux_vec = c2 - c1
    d = aux_vec.dot(norm2) / (veclin_aux.dot(norm2))
    # intersection point
    point_inter = veclin_aux * d + c1

    return point_inter


# use front, back, left, right side seg_image to determine
# the coordinates of bounding cube.
# This function is still not perfect since you need to determine the
# orientation of world coordinate system
def determin_bound_coord(cam_params_arr, seg_img_arr):
    scale_factor = 1.5
    # first determine the upper facet
    xw_set = compute_back_rays(cam_params_arr[0], seg_img_arr[0], direction="top")
    xw_set1 = compute_back_rays(cam_params_arr[1], seg_img_arr[1], direction="top")
    intersection = compute_intersection([cam_params_arr[0], cam_params_arr[1]], [xw_set, xw_set1])
    zmax = intersection[2]

    # next determine the bottom facet
    xw_set = compute_back_rays(cam_params_arr[0], seg_img_arr[0], direction="bottom")
    xw_set1 = compute_back_rays(cam_params_arr[1], seg_img_arr[1], direction="bottom")
    intersection = compute_intersection([cam_params_arr[0], cam_params_arr[1]], [xw_set, xw_set1])
    zmin = intersection[2]

    # then determine the left fact
    xw_set = compute_back_rays(cam_params_arr[0], seg_img_arr[0], direction="left")
    xw_set1 = compute_back_rays(cam_params_arr[1], seg_img_arr[1], direction="right")
    intersection = compute_intersection([cam_params_arr[0], cam_params_arr[1]], [xw_set, xw_set1])
    xmin = intersection[0]

    # determine the right facet
    xw_set = compute_back_rays(cam_params_arr[0], seg_img_arr[0], direction="right")
    xw_set1 = compute_back_rays(cam_params_arr[1], seg_img_arr[1], direction="left")
    intersection = compute_intersection([cam_params_arr[0], cam_params_arr[1]], [xw_set, xw_set1])
    xmax = intersection[0]

    # determine the front facet
    xw_set = compute_back_rays(cam_params_arr[2], seg_img_arr[2], direction="left")
    xw_set1 = compute_back_rays(cam_params_arr[3], seg_img_arr[3], direction="right")
    intersection = compute_intersection([cam_params_arr[2], cam_params_arr[3]], [xw_set, xw_set1])
    ymin = intersection[1]

    # determine the back facet
    xw_set = compute_back_rays(cam_params_arr[2], seg_img_arr[2], direction="right")
    xw_set1 = compute_back_rays(cam_params_arr[3], seg_img_arr[3], direction="left")
    intersection = compute_intersection([cam_params_arr[2], cam_params_arr[3]], [xw_set, xw_set1])
    ymax = intersection[1]

    mid_value = (xmin + xmax) / 2
    xmin = mid_value - scale_factor * (mid_value - xmin)
    xmax = mid_value + scale_factor * (xmax - mid_value)

    mid_value = (ymin + ymax) / 2
    ymin = mid_value - scale_factor * (mid_value - ymin)
    ymax = mid_value + scale_factor * (ymax - mid_value)

    mid_value = (zmin + zmax) / 2
    zmin = mid_value - scale_factor * (mid_value - zmin)
    zmax = mid_value + scale_factor * (zmax - mid_value)

    xmin = round(xmin, 1)
    xmax = round(xmax, 1)
    ymin = round(ymin, 1)
    ymax = round(ymax, 1)
    zmin = round(zmin, 1)
    zmax = round(zmax, 1)

    global limits
    global bound_coord
    limits = [xmin, xmax, ymin, ymax, zmin, zmax]
    bound_coord = [np.array([xmin, ymin, zmax]), np.array([xmax, ymin, zmax]),\
            np.array([xmax, ymax, zmax]), np.array([xmin, ymax, zmax]),\
            np.array([xmin, ymin, zmin]), np.array([xmax, ymin, zmin]),\
            np.array([xmax, ymax, zmin]), np.array([xmin, ymax, zmin])]

    return limits, bound_coord


# initialize the Cartesian grid for computing
# this is a cube-like bounding box instead of shpere or elliptic ball
def init_grid():
    global limits
    # number of point along each axis
    nx = int((limits[1] - limits[0]) / resolution)
    ny = int((limits[3] - limits[2]) / resolution)
    nz = int((limits[5] - limits[4]) / resolution)

    # grid = np.empty((nx, ny, nz, 3), dtype=dt)

    # start = time.clock()
    # this method is 30 times faster than for lop
    x_coord = np.arange(limits[0], limits[1], resolution)
    y_coord = np.arange(limits[2], limits[3], resolution)
    z_coord = np.arange(limits[4], limits[5], resolution)    
    # grid = np.array(np.meshgrid(x_coord, y_coord, z_coord)).T.reshape(-1, 3)

    # due to the realization of meshgrid, I need to exchange x and y axis
    mesh = np.meshgrid(x_coord, y_coord, z_coord)
    x = mesh[0].transpose((1, 0, 2))
    y = mesh[1].transpose((1, 0, 2))
    z = mesh[2].transpose((1, 0, 2))
    grid = np.array((x, y, z))

    # end = time.clock()
    # print(end - start)

    ### actually, due to the realization of meshgrid, here nx is the length along y axis
    ### and ny is the length of x axis
    return grid


# initialize level set function
def init_level_set_function():
    grid = init_grid()
    interface_index, neg_index = init_shape(grid)
    obejct_fmm = fmm.Fmm(grid.shape[1:], interface_index)
    unsigned_phi = obejct_fmm.get_grid()
    sign_mask = np.ones(grid.shape[1:], dtype=np.int16)
    sign_mask[neg_index] = -1
    signed_phi = unsigned_phi * sign_mask
    return interface_index, signed_phi


# initialize the shape, here I used sphere as initial shape
# return the index of the sphere and the index of negative 
# level set
def init_shape(grid):
    c, nx, ny, nz = grid.shape
    radius = int(0.3 * min(nx, ny, nz))
    print("radius is :", radius)

    center_x = int(nx / 2)
    center_y = int(ny / 2)
    center_z = int(nz / 2)

    # start = time.clock()
    temp_phi = (1 << 15) * np.ones((nx, ny, nz), dtype=np.int16)
    # end = time.clock()
    # print("time consumption: ", end - start)

    # generate distance mask
    length_arr = np.arange(-radius, radius + 1, 1) ** 2
    mask = length_arr[:, None, None] + length_arr[None, :, None] + length_arr[None, None, :]

    temp_phi[center_x - radius:center_x + radius + 1,
            center_y - radius:center_y + radius + 1,
            center_z - radius:center_z + radius + 1] = mask

    interface_index = np.where((temp_phi < (radius + 0.5) ** 2) & (temp_phi >= (radius - 0.5) ** 2))
    neg_index = np.where((temp_phi < (radius - 0.5) ** 2))
    return interface_index, neg_index


# get the zero level set index
def get_zero_set_index(grid_SDF):
    return np.where(grid_SDF == 0)


# get negative level set index
def get_neg_set_index(grid_SDF):
    return np.where(grid_SDF < 0)