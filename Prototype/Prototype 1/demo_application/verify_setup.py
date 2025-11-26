"""Verify setup before running"""
from utils.constants import MODEL_PATH, CLASS_NAMES
from utils.text_constants import Texts
from utils.card_game_detector import CardGameDetector
import os

print("Verifying VISIONDECK PRO setup...")
print("-" * 50)
print(f"✓ Model exists: {os.path.exists(MODEL_PATH)}")
print(f"✓ Classes loaded: {len(CLASS_NAMES)}")
print(f"✓ Model path: {MODEL_PATH}")

texts = Texts()
print(f"✓ Title: {texts.get('title')}")

try:
    detector = CardGameDetector(MODEL_PATH, CLASS_NAMES)
    print("✓ Detector initialized successfully")
    print("-" * 50)
    print("All systems ready!")
except Exception as e:
    print(f"✗ Error: {e}")





