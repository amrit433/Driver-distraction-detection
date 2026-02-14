# 🚗 AI-Powered Driver Distraction Detection System

A comprehensive real-time driver monitoring system using **Python**, **OpenCV**, **MediaPipe**, and **Deep Learning (PyTorch)** to detect drowsiness, distraction, yawning, and classify 10 types of driving behaviors.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg)

---

## ✨ Features

### Real-Time Detection
- **👁️ Drowsiness Detection** - Uses Eye Aspect Ratio (EAR) to detect closed/drowsy eyes
- **😴 Yawn Detection** - Uses Mouth Aspect Ratio (MAR) to detect yawning  
- **🔄 Head Pose Estimation** - Detects if driver is looking away from the road
- **🔊 Audio Alerts** - Alarm sounds when distraction/drowsiness detected
- **📊 Real-time Metrics** - Visual display of EAR, MAR, and head pose values

### Deep Learning Classification
- **🧠 10-Class Distraction Classifier** - CNN-based classification using ResNet18/MobileNetV2
- **🎯 99.55% Accuracy** - Trained on State Farm Distracted Driver Dataset
- **⚡ GPU Accelerated** - CUDA support for fast training and inference
- **🔄 Transfer Learning** - Uses ImageNet pretrained weights for better performance

---

## 📋 Distraction Classes

| Class | Code | Description |
|-------|------|-------------|
| c0 | ✅ SAFE | Safe driving |
| c1 | 📱 TEXT_R | Texting (right hand) |
| c2 | 📞 PHONE_R | Phone call (right hand) |
| c3 | 📱 TEXT_L | Texting (left hand) |
| c4 | 📞 PHONE_L | Phone call (left hand) |
| c5 | 📻 RADIO | Operating radio |
| c6 | 🥤 DRINK | Drinking |
| c7 | 🔙 REACH | Reaching behind |
| c8 | 💄 MAKEUP | Hair and makeup |
| c9 | 💬 TALKING | Talking to passenger |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- NVIDIA GPU (optional, for training)

### Setup

1. **Clone/Navigate to the project directory:**
   ```bash
   cd driver-distraction-detection
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **For GPU Training (optional):**
   ```bash
   # Install PyTorch with CUDA support
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

---

## 🚀 Usage

### Easy Run (Windows)

Simply double-click the `run.bat` file in the main project folder. This will automatically activate the environment and start the application.

### Basic Real-Time Detection

```bash
python main.py
```

### With Trained Distraction Classifier

```bash
python main.py --model models/checkpoints/best_model.pth
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | 0 | Camera device ID |
| `--ear` | 0.25 | EAR threshold for drowsiness |
| `--mar` | 0.6 | MAR threshold for yawning |
| `--yaw` | 30 | Head yaw threshold (degrees) |
| `--pitch` | 20 | Head pitch threshold (degrees) |
| `--model` | None | Path to trained distraction classifier |

### Examples

```bash
# Use a different camera
python main.py --camera 1

# More sensitive drowsiness detection
python main.py --ear 0.20

# More sensitive distraction detection  
python main.py --yaw 25 --pitch 15

# Run with deep learning classifier
python main.py --model models/checkpoints/best_model.pth
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `M` | Toggle mesh visualization |
| `R` | Reset counters |

---

## 🏋️ Training Your Own Model

### Step 1: Download Dataset

1. Go to [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)
2. Accept competition rules and download `imgs.zip` (~4GB)
3. Extract to create this structure:
   ```
   data/train/
   ├── c0/   (safe driving)
   ├── c1/   (texting right)
   ├── c2/   (phone right)
   ├── c3/   (texting left)
   ├── c4/   (phone left)
   ├── c5/   (operating radio)
   ├── c6/   (drinking)
   ├── c7/   (reaching behind)
   ├── c8/   (hair/makeup)
   └── c9/   (talking to passenger)
   ```

### Step 2: Train the Model

```bash
# Basic training (20 epochs, ResNet18)
python train_model.py --data data/train --epochs 20

# Faster training with MobileNet
python train_model.py --data data/train --model mobilenet --epochs 10

# Custom settings
python train_model.py --data data/train --batch-size 16 --lr 0.0005
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | data/train | Path to training data |
| `--model` | resnet18 | Model architecture (resnet18/mobilenet) |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--img-size` | 224 | Input image size |
| `--freeze-epochs` | 3 | Epochs to freeze backbone |
| `--workers` | 4 | Data loading workers |
| `--save-dir` | models/checkpoints | Checkpoint save directory |

### Step 3: Run with Trained Model

```bash
python main.py --model models/checkpoints/best_model.pth
```

---

## 🧠 How It Works

### Eye Aspect Ratio (EAR)

The EAR is calculated using 6 eye landmarks to detect eye closure:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

- **Eyes Open**: EAR ≈ 0.3
- **Eyes Closed**: EAR < 0.25 (threshold)

### Mouth Aspect Ratio (MAR)

Similar to EAR, MAR measures mouth openness for yawn detection:

```
MAR = vertical_distance / horizontal_distance
```

- **Mouth Closed**: MAR < 0.5
- **Yawning**: MAR > 0.6 (threshold)

### Head Pose Estimation

Uses OpenCV's `solvePnP` to estimate 3D head orientation from 2D facial landmarks:
- **Yaw**: Left/right rotation
- **Pitch**: Up/down rotation
- **Roll**: Tilt

Detects when the driver is looking away from the road based on angle thresholds.

### Deep Learning Classifier

Uses transfer learning with pretrained CNN architectures:

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|----------|
| ResNet18 | 11.7M | Moderate | 99.55% |
| MobileNetV2 | 3.5M | Fast | ~98% |

Features:
- **Transfer Learning**: Pretrained on ImageNet
- **Fine-tuning**: Backbone frozen for first 3 epochs
- **Data Augmentation**: Random crop, flip, rotation, color jitter
- **Temporal Smoothing**: Averages predictions over 5 frames

---

## 📁 Project Structure

```
driver-distraction-detection/
├── main.py                          # Main application entry point
├── train_model.py                   # Model training script
├── inference.py                     # Real-time prediction module
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                   # PyTorch dataset and dataloaders
│   └── train/                       # Training images (c0-c9 folders)
│
├── models/
│   ├── __init__.py
│   ├── distraction_classifier.py    # ResNet18 & MobileNetV2 models
│   └── checkpoints/
│       ├── best_model.pth           # Best trained model
│       └── latest_model.pth         # Latest checkpoint
│
└── utils/
    ├── __init__.py
    ├── face_detector.py             # MediaPipe face mesh detection
    ├── ear_calculator.py            # Eye Aspect Ratio calculation
    ├── mar_calculator.py            # Mouth Aspect Ratio calculation
    ├── head_pose.py                 # Head pose estimation
    ├── alert_system.py              # Audio/visual alerts
    └── face_landmarker.task         # MediaPipe model file
```

---

## 📊 Training Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | 99.55% |
| **Final Training Accuracy** | 99.62% |
| **Training Time** | ~26 minutes |
| **GPU** | NVIDIA RTX 3050 |
| **Epochs** | 20 |
| **Batch Size** | 32 |

### Training Progress

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 31.78% | 59.53% |
| 4 | 90.66% | 97.86% |
| 10 | 99.09% | 99.26% |
| 20 | 99.62% | 99.55% |

---

## 🔧 Requirements

```
opencv-python
mediapipe
numpy
sounddevice
scipy
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## 💻 System Requirements

- Python 3.8+
- 4GB RAM
- Webcam
- Dual-core CPU


---

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Webcam Issues
```bash
# Try different camera ID
python main.py --camera 1
```

### Low FPS
- Use MobileNet model for faster inference
- Reduce input resolution
- Ensure GPU is being utilized

---

### Quick Start
1. Install dependencies (if not already done):
   
   pip install -r requirements.txt

2. Run the basic real-time detection:
  
   python main.py


3. Run with the trained distraction classifier (recommended):

   python main.py --model models/checkpoints/best_model.pth



This project is for **educational purposes only**. It should **NOT** be relied upon as a safety-critical system. Always drive responsibly and take breaks when tired.


## 🙏 Acknowledgments

- [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) - Dataset
- [MediaPipe](https://mediapipe.dev/) - Face mesh detection
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library

---
