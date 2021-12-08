'''
This code is to visualize a .pcd file using open3d/pptk.
Reference: https://blog.csdn.net/guaiderzhu1314/article/details/105749413
'''
import open3d as o3d
import numpy as np
import os
import pptk
import pdb

def read_pcd(file):
    '''
    Read a pcd file, specifically with X Y Z Intensity
    input: a .pcd file path
    output: a numpy array with [x, y, z, I], shape = (num_points, 4)
    '''
    lines = []
    num_points = None
    with open(file, 'r') as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None
    points = []
    for line in lines[-num_points:]:
        x, y, z, i = list(map(float, line.split()[0:4]))
        points.append(np.array([x, y, z, i]))

    return np.asarray(points)

def visualize_pcd(file, visualize=True, method='open3d'):
    '''
    input:
    file: a .pcd file path
    visualize: a bool to show visualization or not
    method: ['open3d', '']
    '''
    pcd = o3d.io.read_point_cloud(file)
    #pcd = pcl.load(file).to_array()
    print(pcd)
    print(np.asarray(pcd.points))
    print(np.asarray(pcd.colors))
    print(np.asarray(pcd.normals))
    if visualize:
        if method == 'open3d':
            o3d.visualization.draw_geometries([pcd])
        elif method == 'pptk':
            xyz = np.asarray(pcd.points)
            v = pptk.viewer(xyz)
            v.set(point_size=0.005)
            v.wait()
        else:
            print("No such method {}!".format(method))

    return 

def ply2pcd(input_file, output_file):
    '''
    Convert a file from .ply to .pcd
    input_file: a .ply file path
    output_file: a output .pcd file path
    '''
    pcd = o3d.io.read_point_cloud(input_file)
    o3d.io.write_point_cloud(output_file, pcd)

    return

def numpy2pcd(array, output_file):
    '''
    Save a numpy array to a .pcd file (consider xyz only)
    '''
    # create open3d pointcloud object
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(array[:, :3])

    # write to file
    o3d.io.write_point_cloud(output_file, pcl, write_ascii=True, compressed=False, print_progress=False)

    return


if __name__ == '__main__':
    pcd_file1 = '../test_images/Chair.pcd'
    ply_file = '../test_images/Chair.ply'
    pcd_file2 = 'C:\\Users\\lic1syv\\Documents\\project\\shimizu\\data\\extrinsic_calibration\\Nitto_Extrinsic_Test_Rr\\lidar\\000020.pcd'

    # convert a ply file to a pcd file
    #ply2pcd(ply_file, pcd_file)

    # visualize a pcd file
    #visualize_pcd(pcd_file1, True, 'pptk')

    # read a pcd file
    points = read_pcd(pcd_file2)

    # filtering
    #xmin, xmax = 6.0, 13.0
    #xmin, xmax = 29.0, 31.0
    #xmin, xmax = 48.0, 50.0
    xmin, xmax = -47.5, -45
    ymin, ymax = -2.0, 0.3
    zmin, zmax = -2.6, 1.6

    # filtering
    index = np.where((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
                     (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
                     (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
                    )
    points = points[index[0], :]

    name = os.path.basename(pcd_file2)

    numpy2pcd(points[:, :3], os.path.join('./filtered/', name))

    # visualization
    xyz = np.asarray(points[:,:3])
    v = pptk.viewer(xyz)
    v.set(point_size=0.005)
    v.attributes(points[:,-1], points[:, 0])
    v.wait()
