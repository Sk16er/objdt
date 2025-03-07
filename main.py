from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO('yolov8n.pt')

# Configure model parameters
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # NMS IoU threshold
model.classes = None  # filter by class, i.e. = [0, 15, 16] for persons, cats, and dogs

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Run YOLO detection
    results = model(frame, stream=True)  # stream=True for better performance
    
    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence
            conf = box.conf[0]
            
            # Get class name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
