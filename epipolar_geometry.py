import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_essential_matrix(pts1, pts2, K):
    """Computes the Essential Matrix using RANSAC."""
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask

def decompose_essential_matrix(E, pts1, pts2, K):
    """Decomposes Essential Matrix into rotation (R) and translation (t)."""
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def compute_fundamental_matrix(pts1, pts2):
    """Computes the Fundamental Matrix using RANSAC."""
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F

def draw_epipolar_lines(img1, img2, F, pts1, pts2):
    """Draws epipolar lines on both images."""

    def draw_lines(img, lines, pts):
        """Helper function to draw epipolar lines."""
        r, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -line[2] / line[1]])
            x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(pt.astype(int)), 5, color, -1)
        return img

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    img1_lines = draw_lines(img1, lines1, pts1)
    img2_lines = draw_lines(img2, lines2, pts2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img1_lines)
    ax[0].set_title("Epipolar Lines in Image 1")
    ax[1].imshow(img2_lines)
    ax[1].set_title("Epipolar Lines in Image 2")
    # plt.savefig('./results/{}.png', dpi=300)
    plt.show()
