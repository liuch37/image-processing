'''
This code is to visualize a .pcd file using open3d/pptk.
Reference: https://blog.csdn.net/guaiderzhu1314/article/details/105749413
'''
import open3d as o3d
import numpy as np
import pptk
import pdb

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

if __name__ == '__main__':
    pcd_file = '../test_images/Chair.pcd'
    ply_file = '../test_images/Chair.ply'

    # convert a ply file to a pcd file
    #ply2pcd(ply_file, pcd_file)

    # visualize a pcd file
    visualize_pcd(pcd_file, True, 'pptk')
