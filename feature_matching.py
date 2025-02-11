import cv2
import numpy as np

def find_feature_matches(img1, img2, method="SIFT", n_matches=50):
    """
    Finds feature correspondences between two images using the specified method.

    Args:
        img1: First grayscale image.
        img2: Second grayscale image.
        method: Feature detection method ("SIFT", "ORB", "SURF").
        n_matches: Number of matches to retain.

    Returns:
        pts1, pts2: Corresponding points in both images.
        matches: List of matched keypoints.
        keypoints1, keypoints2: Keypoints in both images.
    """

    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=500)
    elif method == "SURF":
        # detector = cv2.xfeatures2d.SURF_create(400)
        raise NotImplementedError("Not implemented yet.")
    else:
        raise ValueError("Invalid method. Choose from 'SIFT', 'ORB', or 'SURF'.")

    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    # Use BFMatcher for matching
    if method == "ORB":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:n_matches]  # Sort and retain best matches

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    return pts1, pts2, matches, keypoints1, keypoints2
