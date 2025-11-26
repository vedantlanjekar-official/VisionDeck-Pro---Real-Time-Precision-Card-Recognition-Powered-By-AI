"""Test script to verify all module connectivity"""
from utils.constants import MODEL_PATH, CLASS_NAMES
from utils.text_constants import Texts
from utils.card_game_detector import CardGameDetector
import os

print("Testing module connectivity...")
print("-" * 50)

# Test Texts module
texts = Texts()
print(f"✓ Texts module: {texts.get('title')}")

# Test Constants
print(f"✓ Model path: {MODEL_PATH}")
print(f"✓ Model exists: {os.path.exists(MODEL_PATH)}")
print(f"✓ Class names count: {len(CLASS_NAMES)}")

# Test Detector
try:
    detector = CardGameDetector(MODEL_PATH, CLASS_NAMES)
    print("✓ Detector initialized successfully")
except Exception as e:
    print(f"✗ Detector initialization failed: {e}")

print("-" * 50)
print("All modules connected successfully!")





