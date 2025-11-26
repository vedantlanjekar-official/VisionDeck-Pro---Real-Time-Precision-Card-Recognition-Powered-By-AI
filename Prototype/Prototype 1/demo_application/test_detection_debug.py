"""Debug script to test card detection"""
from utils.card_game_detector import CardGameDetector
from utils.constants import MODEL_PATH, CLASS_NAMES
import cv2

print("Testing card detection...")
print("-" * 50)

detector = CardGameDetector(MODEL_PATH, CLASS_NAMES)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Capturing test frames...")
all_detections = []

for i in range(10):
    ret, frame = cap.read()
    if ret:
        # Test detection with explicit confidence threshold
        results = detector.model(frame, conf=0.25)
        frame_detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = CLASS_NAMES[cls]
                frame_detections.append(class_name)
                print(f"Frame {i+1}: Detected {class_name} with confidence {conf:.2f}")
        all_detections.extend(frame_detections)
        if not frame_detections:
            print(f"Frame {i+1}: No detections")

cap.release()

print("-" * 50)
print(f"Total detections: {len(all_detections)}")
print(f"Unique detections: {set(all_detections)}")

# Test aggregation
detections = detector.aggregate_detections(all_detections)
print(f"After aggregation (>=3): {detections}")

# Test parsing
parsed_cards = detector.parse_cards(detections)
print(f"Parsed cards: {parsed_cards}")





