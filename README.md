# Visual Odometry

## KITTI Essential Matrix & Feature Matching Analysis

This project processes KITTI odometry dataset images to compute the Essential Matrix, decompose camera motion, and compare results with ground truth. It also visualizes epipolar lines using feature matching techniques.

## 📌 Overview
- Uses KITTI dataset images for camera motion estimation.
- Computes the Essential Matrix using feature correspondences.
- Decomposes the Essential Matrix to obtain rotation and translation.
- Compares results with KITTI ground truth poses.
- Supports `SIFT`and `ORB` feature matching.
<!-- - Visualizes epipolar lines. -->
- Includes a Streamlit web app for interactive analysis.


## 🔧 Installation
### Prerequisites
Ensure you have Python 3.9+ installed along with the following dependencies:
```
pip install -r requirements.txt 
```
Cloning the Repository
```
git clone https://github.com/artzuros/Visual_Odometry.git
cd Visual_Odometry
```
## 🚀 Usage

To run the Streamlit app:
```
streamlit run app.py
```
This opens a web-based UI where you can:

✅ Select the sequence number

✅ Choose the image index

✅ Pick a feature matching method (SIFT or ORB)

✅ See results for Essential Matrix, camera motion, and epipolar lines

OR 

To run the script with different sequences and feature matching methods:


```
python main.py --sequence 00 --image_index 5 --method SIFT
```
Arguments:

- sequence: KITTI sequence number (default: 01).
- image_index: Image index to process (default: 1).
- method: Feature matching method (SIFT, ORB).
- verbose: Prints out all the matrices and relevant information
## 🏗 Setup
Download the KITTI dataset:

- Place the `data_odometry_gray` dataset in the specified path.
- Ensure image and calibration files exist in dataset/sequences/{sequence}/.
- Ensure pose ground truth exists in dataset/poses/{sequence}.txt.

Run the script:

```
python main.py --sequence 01 --image_index 1 --method ORB
```


📂 Project Structure

```
📦 Visual_Odometry
 ┣ 📂 dataset               ┃
 ┃ ┣ 📂 sequences           ┃
 ┃ ┃ ┣ 📂 01                ┃
 ┃ ┃ ┃ ┣ 📂 image_0         ┃-- Download it from https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
 ┃ ┃ ┃ ┣ 📜 calib.txt       ┃
 ┃ ┣ 📂 poses               ┃
 ┃ ┃ ┣ 📜 01.txt            ┃
 ┣ 📜 main.py
 ┣ 📜 calibration.py
 ┣ 📜 feature_matching.py
 ┣ 📜 epipolar_geometry.py
 ┣ 📜 kitti_poses.py
 ┣ 📜 README.md
 ┣ 📜 requirements.txt 
 ┣ 📜 streamlit_app_DEMO.mp4
```
## 📌 TODO
- Check for ever-increasing error on sequence 01 
- Implement real-time visualization of motion estimation.
<!-- Optimize performance for large sequences. -->

<!-- 📝 License
This project is licensed under the MIT License. -->