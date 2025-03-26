# 🏃‍♂️ Human Tracking with YOLO and DeepSORT

## 📌 Overview
Welcome to the **Human Tracking** project! This system leverages **YOLO (You Only Look Once) for object detection** and **DeepSORT for tracking**, enabling real-time human movement tracking using various input sources such as videos, webcams, and wireless cameras. 🎥

## 🚀 Features
- 🔍 **Real-time tracking** with high accuracy
- 📹 **Supports multiple input sources** (Webcam, Videos, Wireless Cameras)
- 🖼 **Generates tracking outputs** (images/videos)
- ⚙️ **Easy to modify and extend**

## 📂 Folder Structure
```
HumanTracking/
│── deep_sort/              # DeepSORT tracking implementation
│── images/                 # Sample images
│── runs/detect/predict/    # Stores prediction output
│── utils/                  # Utility scripts
│── videos/                 # Video samples for testing
│── weights/                # Model weights
│── .gitattributes          # Git configuration
│── HumanPathTracking(yolov8&deepsort).py  # Main tracking script
│── HumanTracking(UsingWebCam).py          # Webcam tracking
│── HumanTracking(usingVideos).py          # Video tracking
│── README.md               # Project documentation
│── output.mp4              # Sample tracking output
│── requirements.txt        # Python dependencies
│── requirements_gpu.txt    # GPU-specific dependencies
│── yolov8n.pt              # YOLO model weights
```

## ⚡ Installation
### 📥 Clone this repository
```sh
 git clone https://github.com/AIvirus/HumanTracking.git
 cd HumanTracking
```
### 📦 Install dependencies
```sh
pip install -r requirements.txt
```
For **GPU support**, install:
```sh
pip install -r requirements_gpu.txt
```
### 🏗 Download YOLO model weights
Place them in the `weights/` folder.

## 🎯 Usage
### ▶️ Tracking from a Video
```sh
python HumanTracking(usingVideos).py --source path_to_video.mp4
```
### 📷 Tracking from a Webcam
```sh
python HumanTracking(UsingWebCam).py
```
### 🎥 Tracking with Wireless Cameras
```sh
python HumanPathTracking(UsingWirelessCam).py
```

## 📊 Output
Tracking results are stored in **`runs/detect/predict/`** as images or video files. 🖼️

## 🔗 Dependencies
- 🐍 Python 3.8+
- 📸 OpenCV
- 🔥 PyTorch
- 🎯 Ultralytics YOLOv8
- 📡 DeepSORT

## 🙌 Acknowledgments
- 🎯 YOLO model by [Ultralytics](https://github.com/ultralytics/yolov5)
- 📌 DeepSORT implementation by [nwojke](https://github.com/nwojke/deep_sort)

## ⚖️ License
🔒 **All rights reserved by the author.**
- 📚 Free to use for educational and research purposes.
- 🚫 Commercial usage requires permission. Reach out if you’d like to discuss collaborations!
