'''
This code is to undistort image given intrinsic calibration.
Reference: https://github.com/vwvw/opencv_live_distortion
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_camera_matrix(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty):
    """
    Retrieve the data from the sliders and put them in the correct matrix format for
    the distortion correction to take place.
    return: a tuple of matrices. First the 3x3 camera matrix and the 14x1
    distortion coefficient matrix. Both matrices are in the same order as
    expected by OpenCV
    """
    distortion_params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty]
    # intrinsic matrix
    mat_intrinsic = np.eye(3)
    mat_intrinsic[0][0] = fx
    mat_intrinsic[1][1] = fy
    mat_intrinsic[0][2] = cx
    mat_intrinsic[1][2] = cy
    # distortion matrix
    mat_dist = np.zeros((14, 1), np.float32)
    for i in range(len(mat_dist)):
        mat_dist[i][0] = distortion_params[i + 4]
    return mat_intrinsic, mat_dist

def undistort(img, camera_mat, dist_mat, method='direct'):
    """
    Retrieve the chosen camera and distortion settings and then undistort the image
    """
    new_m, roi = cv2.getOptimalNewCameraMatrix(
            camera_mat, dist_mat, (img.shape[1], img.shape[0]), 1, (img.shape[1], img.shape[0])
        )
    if method == 'direct':
        dst = cv2.undistort(img, camera_mat, dist_mat, None, new_m)
    elif method == 'remap':
        r_mat = np.eye(3)
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_mat,
            dist_mat,
            r_mat,
            new_m,
            (img.shape[1], img.shape[0]),
            cv2.CV_16SC2,
        )
        dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    else:
        print("Wrong method {} being entered!".format(method))
        exit(1)
    # crop the image
    x, y, w, h = roi
    dst_cropped = dst[y:y+h, x:x+w, :]

    return dst_cropped

# unit testing
if __name__ == '__main__':
    # input path
    test_image_path = '../test_images/distorted_image.png'
    # calibration parameters
    fx = 598.6028602
    fy = 597.89808123
    cx = 648.84050704
    cy = 342.10643555
    k1 = 0.62685222
    k2 = 0.07302406
    p1 = 0.0
    p2 = 0.0
    k3 = 0.00275536
    k4 = 0.98399674
    k5 = 0.20003344
    k6 = 0.01347275
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    tx = 0.0
    ty = 0.0
    # read image
    img = cv2.imread(test_image_path)
    # create matrices
    camera_mat, dist_mat = create_camera_matrix(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty)
    # undistort images
    img_undistorted = undistort(img, camera_mat, dist_mat, 'direct')
    print(img_undistorted)
    print(img_undistorted.shape)
    plt.imshow(img_undistorted)
    plt.show()