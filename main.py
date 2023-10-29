import numpy as np
import open3d
import re
from math import cos, sin

calib_path = "/Users/daniel/Documents/data/KITTI/data_object_calib/training/calib/"
pc_bin_path = "/Users/daniel/Documents/data/KITTI/data_object_velodyne/training/velodyne/"
label_path = "/Users/daniel/Documents/data/KITTI/training/label_2/"

sample_size = 1

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue  
            key, value = re.split(':| ', line, 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def read_label_file(filepath):
    """
    Read in the label file

    return: N x 9 label array where N = # of objects
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue  
            vars = line.split()
            data.append(vars)

    return np.array(data)

def get_box(kitti, calib_path):
    # Output box = [x, y, z, l, w, h, rot]
    inv_matrix = get_inv_matrix(calib_path)
    pos = np.array([float(kitti[11]), float(kitti[12]), float(kitti[13]), 1]).T
    trans_pos = np.matmul(inv_matrix, pos)
    # TODO figure out rotation stuff
    return [trans_pos[0], trans_pos[1], trans_pos[2]+float(kitti[8])/2, float(kitti[10]), float(kitti[9]), float(kitti[8]), -np.pi/2-float(kitti[14])]

def get_inv_matrix(calib_path):
    # TODO From SUSTechPoints
    v2c = "Tr_velo_to_cam"
    rect = "R0_rect"
    with open(calib_path) as file:
        lines = file.readlines()
        trans = [x for x in filter(lambda s: s.startswith(v2c), lines)][0]
        
        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
        matrix = matrix + [0,0,0,1]
        m = np.array(matrix)
        velo_to_cam  = m.reshape([4,4])

        trans = [x for x in filter(lambda s: s.startswith(rect), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]        
        m = np.array(matrix).reshape(3,3)
        
        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)
        
        rect = np.concatenate((m, np.expand_dims(np.array([0,0,0,1]), 0)), axis=0)        
        
        m = np.matmul(rect, velo_to_cam)

        m = np.linalg.inv(m)
        
        return m

def points_in_box(box, points):

    center = box[0:3]
    lwh = box[3:6]
    # TODO Why are we adding 1e-10
    axis_angles = np.array([0, 0, box[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    open3d_pts = open3d.utility.Vector3dVector(points[:,:3])
    ids = box3d.get_point_indices_within_bounding_box(open3d_pts)
    return points[ids]

def visualize(pc_path, calib_path, label_path):

    # read points
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1,4)

    labels = read_label_file(label_path)
    for i, item in enumerate(labels):
        if i != 0:
            labels[i] = float(item)
    
    box = get_box(labels.squeeze(), calib_path) # [x, y, z, l, w, h, rot]
    points = points_in_box(box, points)
    # Plot pcd from bin
    if len(points) == 0:
        return
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    pcd.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3))) # white points

    # Initialize visualization
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0 # smaller points
    vis.get_render_option().background_color = np.zeros(3) # black background
    
    # Add coordinate axis, x forward, y left, z upward 
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pcd)


    vis.run()
    vis.destroy_window()
    
    
    return

for i in range(sample_size):

    name = f'{i}'.zfill(6)
    calib_file = f"{calib_path}{name}.txt"
    label_file = f"{label_path}{name}.txt"
    pc_file = f"{pc_bin_path}{name}.bin"
    print(pc_file)
    visualize(pc_file, calib_file, label_file)

