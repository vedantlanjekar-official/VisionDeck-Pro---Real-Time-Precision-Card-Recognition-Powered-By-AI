# VisionDeck Pro - Running Instructions

## ✅ Setup Complete!

All dependencies have been installed and verified. The project is ready to run.

## 🚀 Running the Applications

### Option 1: Streamlit Application (Belot Score Manager)
A new PowerShell window should have opened automatically with the Streamlit app running.

**If not, run:**
```bash
cd "C:\Users\vedan\Desktop\Projects\VisionDeck Pro\Prototype\Prototype 1\demo_application"
streamlit run main.py
```

**Access the application:**
- Open your web browser
- Go to: `http://localhost:8501`
- The Belot Card Game Tracker interface will load

**Features:**
- Bilingual support (English/Bulgarian)
- Real-time card detection via webcam
- Automatic score calculation for Belot game
- Support for multiple game modes (All Trumps, No Trumps, specific suits)
- Score tracking for two teams

### Option 2: Webcam Detection Application
For real-time card detection visualization:

```bash
cd "C:\Users\vedan\Desktop\Projects\VisionDeck Pro\Prototype\Prototype 1\demo_application"
python model_visualization.py [synthetic|tuned]
```

**Controls:**
- Press `Q` to quit
- Press `S` to toggle confidence labels

**Model Options:**
- `synthetic` - Uses yolov8m_synthetic.pt (52 card classes)
- `tuned` - Uses yolov8m_tuned.pt (13 card classes - hearts only)

### Option 3: Quick Start Batch File
Double-click `start_app.bat` in the project root to start the Streamlit app.

## 📁 Project Structure

- **Backend**: Python with YOLOv8 (Ultralytics)
- **Frontend**: Streamlit web interface
- **Models**: Pre-trained YOLOv8 models in `final_models/`
- **Datasets**: Training datasets in `data/`
- **No Database**: Application uses Streamlit session state (stateless)

## 🔍 Verified Components

✅ Python 3.11.9 installed
✅ All dependencies from requirements.txt installed
✅ Streamlit installed and working
✅ YOLOv8 models verified and accessible
✅ All imports successful
✅ Model paths correctly configured
✅ OpenCV ready for webcam access

## 📝 Model Information

**Available Models:**
1. `yolov8m_synthetic.pt` - 52 card classes, trained on synthetic data
2. `yolov8m_tuned.pt` - 13 card classes (hearts), fine-tuned on real data
3. `yolov8m.pt` - Base model

## 🎮 Usage Tips

1. **Webcam Access**: When running the Streamlit app, allow webcam access when prompted
2. **Card Detection**: Hold playing cards clearly in front of the camera
3. **Lighting**: Ensure good lighting for better detection accuracy
4. **Card Orientation**: Cards should be face-up and clearly visible

## 🛠️ Troubleshooting

If the app doesn't start:
1. Check if port 8501 is already in use
2. Verify webcam is connected and accessible
3. Ensure all model files exist in `final_models/` directory

## 📊 Application Features

### Streamlit App Features:
- Team score tracking
- Card detection from webcam snapshots
- Multiple game mode support
- Bonus points input
- Round history tracking
- Last 10 points tracking

Enjoy using VisionDeck Pro!





