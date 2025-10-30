from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Video Capture
cap = cv2.VideoCapture(0)  # Use webcam
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # Uncomment for video

# Load YOLO model
model = YOLO("../models/yolo8n.pt")

# Class Names (40 example classes, replace these with actual class names from your dataset)
classNames = [
    "person","phone", "cellphone", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup"
]

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    # Calculate FPS
    new_frame_time = time.time()
    success, img = cap.read()  # Capture frame
    if not success:
        break

    results = model(img, stream=True, verbose=False)  # Run YOLO inference

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                cvzone.putTextRect(
                    img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                    scale=1, thickness=3
                )
            else:
                print(f"Invalid class ID: {cls}")

    # FPS Display
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (10, 50), scale=2, thickness=3)

    # Display the frame
    cv2.imshow("YOLO Testing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
