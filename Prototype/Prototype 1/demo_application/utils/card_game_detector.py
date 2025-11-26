import time
from collections import Counter
from ultralytics import YOLO
from utils.game_logic import Card, Suit, Value


class CardGameDetector:
    def __init__(self, model_path, class_names):
        self.model = YOLO(model_path, verbose=False)
        self.class_names = class_names

    def aggregate_detections(self, detections):
        counts = Counter(detections)
        print(counts)
        # Lower threshold from 3 to 1 for better detection
        return [key for key, count in counts.items() if count >= 1]

    def capture_and_process_frames(self, cap, num_frames=10, interval=0.2):
        all_detections = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Use lower confidence threshold for better detection
                results = self.model(frame, conf=0.15, verbose=False)
                frame_detections = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        frame_detections.append(self.class_names[cls])
                all_detections.append(frame_detections)
                time.sleep(interval)
        detections = self.aggregate_detections(all_detections)
        return detections

    def capture_a_frame(self, cap):
        ret, frame = cap.read()
        if ret:
            # Use lower confidence threshold for better detection
            results = self.model(frame, conf=0.15, verbose=False)
            frame_detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    frame_detections.append(self.class_names[cls])
            return frame_detections
        return []
    
    def detect_on_frame(self, frame):
        """Detect cards on a given frame (doesn't read from cap)."""
        # Use lower confidence threshold for better detection
        results = self.model(frame, conf=0.15, verbose=False)
        frame_detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                frame_detections.append(self.class_names[cls])
        return frame_detections

    def parse_card(self, detected_card):
        value = detected_card[:-1]
        suit = detected_card[-1]
        try:
            return Card(Value(value.upper()), Suit(suit))
        except ValueError:
            return None

    def parse_cards(self, detected_cards):
        all_cards = [self.parse_card(card) for card in detected_cards]
        parsed_cards = [parsed_card for parsed_card in all_cards if parsed_card is not None]
        return parsed_cards
