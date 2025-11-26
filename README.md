# VISIONDECK PRO - Playing Cards Object Detection System

<div align="center">

# 🎴 VISIONDECK PRO 🎴

**Real-Time Playing Card Recognition Using YOLOv8**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://github.com/ultralytics/ultralytics)

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [System Architecture](#-system-architecture)
- [ML Modules](#-ml-modules)
- [Datasets](#-datasets)
- [Connectivity](#-connectivity)
- [Quick Start](#-quick-start)

---

## 🎯 Problem Statement

### Challenge

The need for automated card recognition systems that can:
- Accurately identify playing cards from real-world images
- Work in real-time with live video feed
- Handle multiple cards simultaneously
- Provide reliable results for game applications

### Requirements

- Detect all 52 standard playing cards (2-10, J, Q, K, A × 4 suits)
- Real-time processing capability
- High accuracy with minimal false positives
- User-friendly interface for interaction
- Robust detection across different lighting conditions

---

## 💡 Solution

### Approach

**VISIONDECK PRO** is a computer vision application that uses deep learning to solve the card recognition problem:

1. **Deep Learning Model**: Utilizes YOLOv8 (You Only Look Once version 8) object detection model
2. **Pre-trained Models**: Multiple trained models optimized for different scenarios
3. **Real-time Processing**: Live webcam capture with instant detection
4. **Multi-frame Aggregation**: Captures multiple frames and aggregates results for accuracy
5. **Web Interface**: Streamlit-based user interface for easy interaction

### Key Features

- ✅ Real-time card detection from webcam
- ✅ Support for all 52 standard playing cards
- ✅ Optimized detection thresholds (confidence: 0.15, aggregation: 1)
- ✅ Multi-frame validation (10 frames per capture)
- ✅ Automatic results display
- ✅ Simple and intuitive user interface

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│              User Interface Layer                    │
│              (Streamlit Web Application)             │
│                                                      │
│  • Web Interface (main.py)                          │
│  • User Interaction Handling                        │
│  • State Management                                 │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│            Application Logic Layer                   │
│                                                      │
│  • Card Capture (capture_cards)                     │
│  • Results Display (display_results)                │
│  • Webcam Management (OpenCV)                       │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│           Detection Engine Layer                     │
│       (CardGameDetector - card_game_detector.py)    │
│                                                      │
│  • Frame Processing                                 │
│  • Detection Aggregation                            │
│  • Card Parsing                                     │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│            Machine Learning Layer                    │
│              (YOLOv8 Model - Ultralytics)           │
│                                                      │
│  • Object Detection Inference                       │
│  • Bounding Box Prediction                          │
│  • Class Classification (52 classes)                │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              Hardware Layer                          │
│                                                      │
│  • Webcam (OpenCV VideoCapture)                     │
│  • GPU (CUDA - Optional)                            │
└─────────────────────────────────────────────────────┘
```

### Component Description

#### 1. User Interface Layer
- **Technology**: Streamlit
- **File**: `demo_application/main.py`
- **Responsibilities**:
  - Render web interface
  - Handle user interactions
  - Manage session state
  - Display detection results

#### 2. Application Logic Layer
- **Technology**: Python, OpenCV
- **File**: `demo_application/main.py`
- **Responsibilities**:
  - Webcam initialization and management
  - Frame capture coordination
  - Results presentation
  - User workflow management

#### 3. Detection Engine Layer
- **Technology**: Python, Collections
- **File**: `demo_application/utils/card_game_detector.py`
- **Responsibilities**:
  - Frame-by-frame detection
  - Multi-frame aggregation
  - Card string parsing
  - Detection validation

#### 4. Machine Learning Layer
- **Technology**: YOLOv8 (Ultralytics), PyTorch
- **Model File**: `final_models/yolov8m_synthetic.pt`
- **Responsibilities**:
  - Image preprocessing
  - Object detection inference
  - Class prediction
  - Confidence scoring

#### 5. Hardware Layer
- **Technology**: OpenCV, CUDA (optional)
- **Responsibilities**:
  - Video capture from webcam
  - Frame acquisition
  - GPU acceleration (if available)

### Data Flow

```
1. User clicks "Take Snapshot"
   ↓
2. Webcam captures 10 frames (640x480)
   ↓
3. Each frame sent to YOLOv8 model
   ↓
4. Model returns detections (card classes + confidence)
   ↓
5. Detections aggregated across frames
   ↓
6. Cards filtered by aggregation threshold (≥1)
   ↓
7. Card strings parsed to Card objects
   ↓
8. Results displayed in Streamlit interface
```

---

## 🤖 ML Modules

### Machine Learning Components

#### 1. YOLOv8 Model Architecture

**Model Type**: YOLOv8 Medium (YOLOv8m)
- **Architecture**: CSPDarknet53 backbone + PANet neck + Detection head
- **Input Size**: 640x640 pixels
- **Output**: Bounding boxes + Class probabilities
- **Number of Classes**: 52 (all standard playing cards)

#### 2. CardGameDetector Class

**Location**: `demo_application/utils/card_game_detector.py`

**Key Methods**:

```python
class CardGameDetector:
    def __init__(self, model_path, class_names):
        """Initialize detector with model and class names"""
        
    def detect_on_frame(self, frame):
        """Detect cards in a single frame"""
        # Uses confidence threshold: 0.15
        # Returns: List of detected card class names
        
    def aggregate_detections(self, detections):
        """Aggregate detections across multiple frames"""
        # Counts occurrences
        # Filters by threshold: ≥1 detection
        # Returns: List of unique card names
        
    def parse_cards(self, detected_cards):
        """Convert card strings to Card objects"""
        # Parses format: "10h" → Card(value=10, suit=hearts)
        # Returns: List of Card objects
```

**Detection Parameters**:
- **Confidence Threshold**: 0.15 (15% minimum confidence)
- **Aggregation Threshold**: 1 (card must appear at least once)
- **Frames Captured**: 10 frames per snapshot
- **Frame Resolution**: 640x480 pixels

#### 3. Model Files

| Model File | Classes | Size | Use Case |
|------------|---------|------|----------|
| `yolov8m_synthetic.pt` | 52 | ~50 MB | Production (Default) |
| `yolov8m_tuned.pt` | 13 | ~50 MB | Hearts suit only |
| `yolov8m.pt` | Base model | ~50 MB | Base YOLOv8 model |

#### 4. Inference Process

```python
# Step 1: Load Model
model = YOLO('yolov8m_synthetic.pt')

# Step 2: Process Frame
results = model(frame, conf=0.15)

# Step 3: Extract Detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    card_name = CLASS_NAMES[class_id]
```

---

## 📊 Datasets

### Available Datasets

The project includes four datasets organized in the `data/` directory:

#### 1. Synthetic Dataset
- **Location**: `data/synthetic_dataset/`
- **Total Images**: 20,000
- **Classes**: 52 (all standard playing cards)
- **Split**:
  - Train: 14,000 images
  - Validation: 4,000 images
  - Test: 2,000 images
- **Source**: Kaggle - Playing Cards Object Detection Dataset
- **Format**: YOLOv8 format (images + labels)

#### 2. Real Dataset
- **Location**: `data/real_dataset/`
- **Total Images**: 100
- **Classes**: 13 (hearts suit: 2h-10h, Jh, Qh, Kh, Ah)
- **Split**:
  - Train: 69 images
  - Validation: 18 images
  - Test: 11 images
- **Format**: YOLOv8 format

#### 3. Real Augmented Dataset
- **Location**: `data/real_augmented_dataset/`
- **Total Images**: 1,000 (10x augmentation of real dataset)
- **Classes**: 13 (hearts suit)
- **Augmentation**: Rotation, brightness, contrast adjustments
- **Split**:
  - Train: 690 images
  - Validation: 180 images
  - Test: 110 images

#### 4. Combined Dataset
- **Location**: `data/combined_dataset/`
- **Total Images**: 1,100
- **Classes**: 13 (hearts suit)
- **Composition**: Real dataset + subset of synthetic dataset
- **Split**:
  - Train: 759 images
  - Validation: 198 images
  - Test: 121 images

### Dataset Structure

All datasets follow YOLOv8 standard format:

```
dataset_name/
├── train/
│   ├── images/          # Training images (.jpg)
│   └── labels/          # YOLO annotations (.txt)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # YOLO annotations
├── test/
│   ├── images/          # Test images
│   └── labels/          # YOLO annotations
└── data.yaml            # Dataset configuration
```

### YOLO Annotation Format

Each annotation file contains normalized bounding box coordinates:

```
class_id center_x center_y width height
```

Example:
```
0 0.5 0.5 0.2 0.3    # Card at center, 20% width, 30% height
12 0.3 0.7 0.15 0.25 # Another card
```

All coordinates are normalized (0.0 to 1.0).

---

## 🔌 Connectivity

### Module Connectivity Overview

```
main.py
  ├── imports card_game_detector.py
  ├── imports constants.py (MODEL_PATH, CLASS_NAMES)
  ├── imports text_constants.py (Texts)
  └── uses OpenCV (cv2) for webcam

card_game_detector.py
  ├── uses YOLOv8 (ultralytics)
  ├── uses game_logic.py (Card, Suit, Value classes)
  └── uses constants.py (CLASS_NAMES)

constants.py
  └── defines MODEL_PATH and CLASS_NAMES

text_constants.py
  └── defines UI text strings (English only)
```

### Detailed Module Connections

#### 1. Frontend to Backend

**File**: `demo_application/main.py`

```python
# Imports
from utils.card_game_detector import CardGameDetector  # Detection engine
from utils.constants import MODEL_PATH, CLASS_NAMES    # Configuration
from utils.text_constants import Texts                 # UI texts
import cv2                                             # Webcam access

# Connection Flow
detector = CardGameDetector(MODEL_PATH, CLASS_NAMES)  # Initialize detector
# → Loads YOLOv8 model from MODEL_PATH
# → Uses CLASS_NAMES for mapping detections
```

#### 2. Detection Engine

**File**: `demo_application/utils/card_game_detector.py`

```python
# Imports
from ultralytics import YOLO              # YOLOv8 framework
from utils.game_logic import Card, Suit, Value  # Card data structures

# Model Loading
self.model = YOLO(model_path)            # Loads PyTorch model
# → Connects to final_models/yolov8m_synthetic.pt

# Detection Flow
results = self.model(frame, conf=0.15)   # Inference
# → Processes frame through YOLOv8
# → Returns bounding boxes and classes
```

#### 3. Configuration Connection

**File**: `demo_application/utils/constants.py`

```python
MODEL_PATH = "../final_models/yolov8m_synthetic.pt"
# → Points to trained model file

CLASS_NAMES = ["10c", "10d", ...]  # 52 classes
# → Maps class IDs (0-51) to card names
```

#### 4. Webcam Connection

**File**: `demo_application/main.py`

```python
cap = cv2.VideoCapture(0)
# → Opens default webcam (index 0)
# → Sets resolution to 640x480
# → Provides frame stream
```

### Data Flow Between Modules

```
┌─────────────┐
│   main.py   │
│             │
│ 1. User     │
│    Action   │
└──────┬──────┘
       │
       │ calls
       ▼
┌──────────────────────────┐
│ CardGameDetector         │
│                          │
│ 2. detect_on_frame()     │
│    → processes frame     │
└──────┬───────────────────┘
       │
       │ uses
       ▼
┌──────────────────────────┐
│ YOLOv8 Model             │
│                          │
│ 3. Model Inference       │
│    → returns detections  │
└──────┬───────────────────┘
       │
       │ maps to
       ▼
┌──────────────────────────┐
│ CLASS_NAMES              │
│ (constants.py)           │
│                          │
│ 4. Class ID → Card Name  │
│    (0 → "10c", etc.)     │
└──────┬───────────────────┘
       │
       │ aggregates
       ▼
┌──────────────────────────┐
│ aggregate_detections()   │
│                          │
│ 5. Count & Filter        │
│    → final card list     │
└──────┬───────────────────┘
       │
       │ parses
       ▼
┌──────────────────────────┐
│ parse_cards()            │
│                          │
│ 6. String → Card Object  │
│    ("10h" → Card obj)    │
└──────┬───────────────────┘
       │
       │ displays
       ▼
┌─────────────┐
│   main.py   │
│             │
│ 7. Show     │
│    Results  │
└─────────────┘
```

### Dependency Graph

```
main.py
  ├── streamlit (UI framework)
  ├── cv2 (webcam)
  ├── utils.card_game_detector
  │     ├── ultralytics.YOLO
  │     ├── utils.game_logic
  │     └── collections.Counter
  ├── utils.constants
  └── utils.text_constants
```

### File Dependencies

| Module | Dependencies | Purpose |
|--------|--------------|---------|
| `main.py` | Streamlit, OpenCV, CardGameDetector, constants, text_constants | Main application entry point |
| `card_game_detector.py` | YOLOv8, game_logic, Counter | Detection engine |
| `constants.py` | None (configuration) | Model paths and class definitions |
| `text_constants.py` | None (configuration) | UI text strings |

---

## 🚀 Quick Start

### Installation

```bash
# 1. Navigate to project directory
cd "VisionDeck Pro/Prototype/Prototype 1"

# 2. Install dependencies
pip install -r requirements.txt
pip install streamlit

# 3. Verify setup
python demo_application/verify_setup.py
```

### Running the Application

```bash
# Navigate to demo application
cd demo_application

# Start Streamlit
streamlit run main.py
```

### Access

Open browser to: `http://localhost:8501`

---

## 📝 Project Structure

```
Prototype/Prototype 1/
├── demo_application/           # Main application
│   ├── main.py                 # Streamlit web app
│   ├── utils/
│   │   ├── card_game_detector.py    # Detection engine
│   │   ├── constants.py             # Model configuration
│   │   ├── game_logic.py            # Card data structures
│   │   └── text_constants.py        # UI texts
│   └── model_visualization.py # CLI detection tool
│
├── final_models/               # Pre-trained models
│   ├── yolov8m_synthetic.pt   # Main model (52 classes)
│   ├── yolov8m_tuned.pt       # Fine-tuned (13 classes)
│   └── yolov8m.pt             # Base model
│
├── data/                       # Training datasets
│   ├── synthetic_dataset/     # 20,000 images
│   ├── real_dataset/          # 100 images
│   ├── real_augmented_dataset/# 1,000 images
│   └── combined_dataset/      # 1,100 images
│
├── model_utils/                # Training utilities
│   ├── train.py
│   ├── val.py
│   └── predict.py
│
└── requirements.txt            # Dependencies
```

---

## 🔧 Configuration

### Model Configuration

**File**: `demo_application/utils/constants.py`

```python
MODEL_PATH = "../final_models/yolov8m_synthetic.pt"  # Model file path
CLASS_NAMES = ["10c", "10d", ...]  # 52 card classes
```

### Detection Configuration

**File**: `demo_application/utils/card_game_detector.py`

- **Confidence Threshold**: 0.15
- **Aggregation Threshold**: 1
- **Number of Frames**: 10

---

## 📊 Technical Specifications

### Model Details
- **Architecture**: YOLOv8 Medium
- **Input Size**: 640x640 pixels
- **Classes**: 52 (all standard playing cards)
- **Format**: PyTorch (.pt)

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Webcam**: USB webcam (640x480 or higher)
- **GPU**: Optional (CUDA support for faster inference)

### Performance
- **Inference Time**: ~50ms per frame (CPU), ~10ms (GPU)
- **Accuracy**: 98-99% for single cards
- **Detection Rate**: Supports 1-5 cards per capture

---

<div align="center">

**VISIONDECK PRO** - Real-Time Card Recognition System

Built with YOLOv8 and Streamlit

</div>
