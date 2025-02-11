import numpy as np

def read_kitti_poses(pose_file):
    """Reads KITTI ground truth poses and returns a list of (R, t) pairs."""
    poses = []
    with open(pose_file, "r") as f:
        for line in f.readlines():
            data = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
            R = data[:, :3]
            t = data[:, 3]
            poses.append((R, t))
    return poses
