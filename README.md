# farmers-efficiency-jetson-nano

This project aims to enhance farming efficiency using computer vision and deep learning techniques implemented on a Jetson Nano device. The project consists of three main components: weed detection using YOLOv4 and YOLOv5, and lane detection using Hough Transform.

## Project Components

### 1. YOLOv4 Weed Detection
Located in `yolov4-Weed-Detection/`, this component uses YOLOv4 for real-time weed detection in agricultural fields.
- Features:
  - Image-based weed detection (`detect_image.py`)
  - Video-based weed detection (`detect_video.py`)
  - Pre-trained weights included (`crop_weed_detection.weights`)
  - Configuration file for model architecture (`crop_weed.cfg`)
- Note: This implementation runs on CPU as it doesn't support GPU acceleration on Jetson Nano

### 2. YOLOv5 Weed Detection
Located in `yolov5-Weed-Detection/`, this is an improved version using YOLOv5 for weed detection with GPU acceleration support.
- Features:
  - Real-time detection capabilities (`detectRT.ipynb`)
  - ONNX model support for optimized inference (`detect_using_onnx.ipynb`)
  - Image and video detection scripts
  - Pre-trained ONNX model included (`best.onnx`)
  - TensorRT optimization for GPU acceleration on Jetson Nano
  - Significantly improved inference speed compared to YOLOv4 implementation
  - Hardware-accelerated inference leveraging Jetson Nano's GPU capabilities

### 3. Lane Detection
Located in `Hough-transform-laneDetection/`, this component implements lane detection using Hough Transform for agricultural vehicle navigation.
- Features:
  - Lane detection in agricultural fields
  - Jupyter notebook implementation for easy experimentation

## Setup Instructions

### Prerequisites
- Jetson Nano device
- Python 3.x
- CUDA support (for GPU acceleration)
- Necessary Libraries such as: OpenCV, TensorRT (for YOLOv5 GPU optimization), Matplotlib and NumPy

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd jetson-nano-enhance-farmers-efficiency
```
2. Setup the environment with required dependencies on jetson nano.

## Usage

### YOLOv4 Weed Detection
```bash
# For image detection
python detect_image.py 

# For video detection
python detect_video.py 
```

### YOLOv5 Weed Detection
```bash
# For real-time detection with GPU acceleration
python detectRT.ipynb

# For ONNX model inference with TensorRT optimization
python detect_using_onnx.ipynb
```

### Lane Detection
Open and run the Jupyter notebook in the Hough-transform-laneDetection directory.

## Contributors
- [Soham Kukreti](https://github.com/SohamKukreti)
- [Yuvraj Rathi](https://github.com/yryuvraj)
- [Satyam Rathi](https://github.com/satyamrathirar)
