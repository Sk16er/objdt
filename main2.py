from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('yolov8n.pt')  # load the smallest YOLOv8 model

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Run YOLO detection
    results = model(frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()