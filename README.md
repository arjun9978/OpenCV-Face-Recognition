# Face Detection & Recognition Using OpenCV

## Overview

This project implements face detection and recognition using OpenCV. It utilizes Haar Cascade Classifiers for face detection and LBPH (Local Binary Pattern Histogram) for face recognition. The system can train on face datasets, recognize individuals in real time, and detect faces efficiently.

## Features

Face Detection using Haar Cascade Classifiers.

Face Recognition using LBPH (Local Binary Pattern Histogram) algorithm.

Real-time Processing with OpenCV and a webcam.

Training Model Storage for recognition.

## Technologies Used

Python

OpenCV for image processing

NumPy for numerical operations

PIL (Pillow) for image handling

Haar Cascade for face detection

LBPH (Local Binary Pattern Histogram) for face recognition

## Installation

- Clone the repository

- Install dependencies

pip install opencv-python numpy pillow

## Usage

- 1. Run the databse file under Face Recognition folder

A dataset of face images is generated and stored in the dataset/ directory. Then, run the training script:

python train.py

This will train the model and save the trained file in the trainer/ folder.

- 2. Recognize Faces

Run the real-time recognition script:

python recognize.py

This will open a webcam feed and recognize faces based on the trained model.

## Folder Structure

face-recognition-opencv/
│── dataset.py                # Face dataset script
│── train.py                # Face training script
│── recognize.py            # Face recognition script
│── detect.py               # Face detection script
│── haarcascade_frontalface_default.xml  # Haar Cascade model
│── README.md               # Project documentation

## Notes

- Ensure the dataset contains clear face images in different angles under good face light

= Adjust Haar Cascade parameters if detection is inaccurate.

- LBPH is effective for small datasets but may struggle with large variations in lighting and pose.

## License

This project is open-source and licensed under the MIT License.

