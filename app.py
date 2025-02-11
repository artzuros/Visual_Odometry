import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibration import read_calibration
from feature_matching import find_feature_matches
from epipolar_geometry import compute_essential_matrix, decompose_essential_matrix, compute_fundamental_matrix, draw_epipolar_lines
from kitti_poses import read_kitti_poses

# Streamlit UI
st.title("KITTI Visual Odometry")
st.sidebar.header("Input Parameters")

# User Inputs
data_path = st.sidebar.text_input("Path to KITTI Dataset", "/home/artzuros/Documents/CS/kitti/data_odometry_gray/dataset/")
sequence = st.sidebar.text_input("KITTI Sequence Number", "01")
image_number = st.sidebar.number_input("Image Number", min_value=1, step=1, value=1)
method = st.sidebar.selectbox("Feature Matching Method", ["SIFT", "ORB"])

# Construct Paths
image_path = f"{data_path}/sequences/{sequence}/image_0/"
calib_file = f"{data_path}/sequences/{sequence}/calib.txt"
pose_file = f"{data_path}/poses/{sequence}.txt"

# Load images
img1_path = f"{image_path}{image_number:06d}.png"
img2_path = f"{image_path}{image_number+1:06d}.png"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    st.error("Could not load images. Check the dataset path, sequence, and image number.")
else:
    # Load calibration data
    K = read_calibration(calib_file)

    # Find feature correspondences
    pts1, pts2, matches, keypoints1, keypoints2 = find_feature_matches(img1, img2, method=method)

    # Compute Essential Matrix
    E, _ = compute_essential_matrix(pts1, pts2, K)
    R, t = decompose_essential_matrix(E, pts1, pts2, K)

    # Load KITTI ground truth poses
    poses = read_kitti_poses(pose_file)
    R_gt, t_gt = poses[image_number + 1]
    print(R_gt, t_gt)
    # Compute errors
    t_gt /= np.linalg.norm(t_gt)
    t /= np.linalg.norm(t)
    rotation_error = np.arccos((np.trace(R_gt @ R.T) - 1) / 2) * (180 / np.pi)
    translation_error = np.arccos(np.clip(np.dot(t_gt, t), -1.0, 1.0)) * (180 / np.pi)

    # Display results
    st.subheader("Feature Matching Results")
    fig, ax = plt.subplots(figsize=(20, 10))
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None)
    ax.imshow(img_matches, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Computed Matrices")
    st.text(f"Essential Matrix:\n{np.array2string(E, precision=4, suppress_small=True)}")
    st.text(f"Rotation Matrix:\n{np.array2string(R, precision=4, suppress_small=True)}")
    st.text(f"Translation Vector:\n{np.array2string(t, precision=4, suppress_small=True)}")

    st.subheader("Errors")
    st.write(f"Rotation Error: {rotation_error:.4f} degrees")
    st.write(f"Translation Error: {translation_error:} degrees")

    # # Compute and visualize epipolar lines
    # st.subheader("Epipolar Geometry")
    # F = compute_fundamental_matrix(pts1, pts2)
    
    # fig_epi = draw_epipolar_lines(img1, img2, F, pts1[:10], pts2[:10])
    # st.pyplot(fig_epi)  # Display epipolar lines plot
