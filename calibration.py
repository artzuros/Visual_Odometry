import numpy as np

def read_calibration(calib_file, rgb=False):
    """Reads KITTI calibration file and extracts intrinsic matrix."""
    with open(calib_file, "r") as f:
        lines = f.readlines()

    # Extract P0 (left camera projection matrix)
    if rgb:
        P2 = np.array([float(val) for val in lines[2].split(":")[1].split()]).reshape(3, 4)
        K = P0[:, :3]
    else:
        P0 = np.array([float(val) for val in lines[0].split(":")[1].split()]).reshape(3, 4)
        # Intrinsic matrix K (first 3x3 of P0)
        K = P0[:, :3]

    return K
