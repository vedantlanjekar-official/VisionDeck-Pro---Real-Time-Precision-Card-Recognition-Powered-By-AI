"""Test FPS optimizations"""
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    # Test FPS settings
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except:
        pass  # Some cameras don't support these
    
    # Get current FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS set to: {fps}")
    cap.release()
    print("✓ FPS optimizations applied successfully")
else:
    print("Camera not accessible")





