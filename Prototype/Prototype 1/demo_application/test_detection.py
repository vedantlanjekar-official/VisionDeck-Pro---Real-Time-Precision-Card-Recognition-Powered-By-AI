"""Test card detection"""
from utils.card_game_detector import CardGameDetector
from utils.constants import MODEL_PATH, CLASS_NAMES
import cv2

detector = CardGameDetector(MODEL_PATH, CLASS_NAMES)
print("Detector initialized")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
else:
    ret, frame = cap.read()
    cap.release()
    if ret:
        print("Frame captured")
        detections = detector.detect_on_frame(frame)
        print(f"Detection method works! Found {len(detections)} detections")
        if detections:
            print(f"Detected: {detections}")
    else:
        print("Error: Could not read frame")





