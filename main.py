import cv2
import datetime
import numpy as np
import argparse
from calibration import read_calibration
from feature_matching import find_feature_matches
from epipolar_geometry import compute_essential_matrix, decompose_essential_matrix, compute_fundamental_matrix, draw_epipolar_lines
from kitti_poses import read_kitti_poses

def run_kitti_analysis(sequence="01", image_index=1, method="ORB", verbose=False):
    # Load images
    image_path = f"{DATASET}/sequences/{sequence}/image_0/"
    img1 = cv2.imread(f"{image_path}{image_index:06d}.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f"{image_path}{image_index + 1:06d}.png", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check file paths.")

    # Load calibration data
    K = read_calibration(f"{DATASET}/sequences/{sequence}/calib.txt")
    
    if verbose:
        print("\n📌 Camera Intrinsic Matrix (K):\n", K)

    start = datetime.datetime.now()
    # Find feature correspondences
    pts1, pts2, matches, keypoints1, keypoints2 = find_feature_matches(img1, img2, method=method)
    print(f"\nFeature matching took: {datetime.datetime.now() - start}")

    # Compute and decompose Essential Matrix
    E, _ = compute_essential_matrix(pts1, pts2, K)
    R, t = decompose_essential_matrix(E, pts1, pts2, K)

    if verbose:
        print("\n📌 Essential Matrix (E):\n", E)
        print("\n📌 Estimated Rotation Matrix (R):\n", R)
        print("\n📌 Estimated Translation Vector (t):\n", t)

    # Load KITTI ground truth poses
    pose_file = f"{DATASET}/poses/{sequence}.txt"
    poses = read_kitti_poses(pose_file)
    R_gt, t_gt = poses[image_index + 1]

    if verbose:
        print("\n📌 Ground Truth Rotation Matrix (R_gt):\n", R_gt)
        print("\n📌 Ground Truth Translation Vector (t_gt):\n", t_gt)

    # Compute errors
    t_gt /= np.linalg.norm(t_gt)
    t /= np.linalg.norm(t)
    rotation_error = np.arccos(np.clip((np.trace(R_gt @ R.T) - 1) / 2, -1.0, 1.0)) * (180 / np.pi)
    translation_error = np.arccos(np.clip(np.dot(t_gt, t), -1.0, 1.0)) * (180 / np.pi)

    print(f"\n📌 Rotation Error: {rotation_error:.4f} degrees")
    print(f"📌 Translation Error: {translation_error} degrees")

    # Compute and visualize epipolar lines
    F = compute_fundamental_matrix(pts1, pts2)

    if verbose:
        print("\n📌 Fundamental Matrix (F):\n", F)

    draw_epipolar_lines(img1, img2, F, pts1[:10], pts2[:10])

if __name__ == "__main__":
    DATASET = '/home/artzuros/Documents/CS/kitti/data_odometry_gray/dataset'
    parser = argparse.ArgumentParser(description="Run KITTI essential matrix and feature matching analysis.")
    parser.add_argument("--sequence", type=str, default="01", help="KITTI sequence number")
    parser.add_argument("--image_index", type=int, default=1, help="Index of the first image to use")
    parser.add_argument("--method", type=str, default="ORB", choices=["SIFT", "ORB", "SURF"], help="Feature matching method")
    parser.add_argument("--verbose", action="store_true", help="Print all computed matrices")

    args = parser.parse_args()
    run_kitti_analysis(sequence=args.sequence, image_index=args.image_index, method=args.method, verbose=args.verbose)
