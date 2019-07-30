import numpy as np
import cv2
import glob


num_cams = 36


class CameraParams:
    def __init__(self, R, t, K):
        self.__R = R
        # t here is the camera origin point w.r.t world coordinate system
        self.__t = t
        self.__K = K

    def getRotation(self):
        return self.__R

    def getMotion(self):
        return self.__t

    def getIntrinsic(self):
        return self.__K


class ImageData:
    def __init__(self, folder_path, method=1):
        self.__data = []

        # first get the all the paths of files
        files_paths = sorted(glob.glob(folder_path + "/*.jpg"))
        for i in files_paths:
            data = cv2.imread(i, method)
            self.__data.append(data)

    def get_data(self):
        return self.__data


# return the parameter of camera parameters
def get_cam_param(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print("File not open")
    # intrinsic matrix
    all_cam_param = []
    for i in range(num_cams):
        s_index = str(i).zfill(3)
        # projection matrix
        P = fs.getNode("viff" + s_index + "_matrix").mat()
        K, R, t = cv2.decomposeProjectionMatrix(P)[0:3]
        t = np.squeeze(t)
        np.divide(t, t[3], out=t)
        # print(T.shape)
        t = t[0:-1]
        cam_params = CameraParams(R, t, K)
        all_cam_param.append(cam_params)
    # print(array_T)
    fs.release()
    # fs = cv2.FileStorage("squirrel.yml", cv2.FILE_STORAGE_WRITE)
    # fs.write("Rotations", "[")
    # fs.write("Rotations", np.array(array_R))
    # fs.write("Rotations", "]")
    # fs.write("Motions", np.array(array_T))
    # fs.release()
    return all_cam_param


if __name__ == "__main__":
    get_cam_param("viff.xml")
