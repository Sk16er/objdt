# OBJDT

This Python project implements real-time object detection using the YOLOv8 (You Only Look Once) model. It captures video from your webcam, detects objects in the live feed, and highlights them with bounding boxes. The detected object's class name and confidence score are displayed on the screen.

## Requirements

Before running the script, make sure you have the following dependencies installed:

- **Python 3.x**: The code is written in Python, so you need Python 3 or above.
- **OpenCV**: Used for webcam capture, frame processing, and displaying results.
- **Ultralytics YOLOv8**: The pre-trained YOLOv8 model is used for object detection.
- **NumPy**: Required by OpenCV for array manipulations.

### Installing Dependencies

To set up the environment, install the required libraries by running the following command:

```bash
pip install opencv-python-headless numpy ultralytics
```


> python main.py


> python main2.py


> model.conf = 0.25  # Confidence threshold


### Explanation of the `README.md`:

- **Project Overview**: It starts by explaining the core functionality of the project, which is to perform real-time object detection with YOLOv8 using a webcam.
- **Requirements**: Lists the libraries needed for the project (OpenCV, NumPy, and YOLOv8).
- **Installing Dependencies**: Provides installation instructions to install necessary dependencies.
- **YOLOv8 Model**: Informs the user about the required model file (`yolov8n.pt`).
- **Usage**: Explains how to run the script and details what happens during the execution.
- **Customization**: Provides information about how to customize key parameters, including confidence threshold, IoU threshold, and class filtering.
- **Troubleshooting**: Provides solutions for common problems (camera access issues, model loading, performance issues).
- **License**: Mentions that the project is licensed under the MIT License (you can change this depending on your needs).

This `README.md` will provide clear guidance for anyone setting up or using the project. Let me know if you need any further modifications or additional details!



