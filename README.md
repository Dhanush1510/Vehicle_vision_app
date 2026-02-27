# ML-BASED REAL-TIME VIDEO ANALYTICS WITH MULTI-MODE VISUAL ENHANCEMENT

## Project Overview

This project presents a real-time computer vision framework for performing object detection on user-uploaded videos after applying advanced visual enhancement techniques.

Using Flask for web deployment, OpenCV for video processing, classical computer vision algorithms for visual enhancement, and YOLOv8 for deep learning–based object detection, the system simulates multi-sensor perception environments such as LIDAR, thermal imaging, and fog-robust vision.

The framework enables dynamic streaming of annotated frames through a web interface, demonstrating the integration of image processing and deep learning for intelligent video analytics.

---

## Objective

To develop a systematic and real-time video analytics pipeline that:

- Accepts user-uploaded video input
- Applies sensor-inspired visual enhancement techniques
- Performs deep learning–based object detection
- Streams annotated frames dynamically via a web interface
- Simulates multi-sensor perception using computer vision methods

---

## Methodology

### 1. Data Input

- User uploads a video file through the Flask web interface
- The video is stored temporarily using Python tempfile
- OpenCV extracts frames sequentially for processing

### 2. Visual Enhancement Modes

For each frame extracted from the uploaded video, one of the following enhancement techniques is applied:

#### LIDAR Simulation

- Grayscale conversion
- Bilateral filtering for noise reduction
- Intensity-based color mapping:
  - Blue for distant objects
  - Purple for mid-far objects
  - Magenta for mid-range objects
  - Orange for near objects
- Randomized point-cloud masking
- Gaussian blur smoothing

This simulates a depth-based LIDAR visualization effect.

#### Thermal Imaging Simulation

- Grayscale conversion
- Application of OpenCV JET colormap
- Contrast enhancement using intensity scaling

This simulates infrared-style thermal visualization.

#### Image Dehazing (Dark Channel Prior)

- Dark channel extraction
- Atmospheric light estimation
- Transmission map estimation
- Guided filtering for refinement
- Scene radiance recovery
- CLAHE-based contrast enhancement

This improves visibility under foggy or hazy environmental conditions.

### 3. Object Detection

After enhancement:

- Frames are passed into YOLOv8 (yolov8s.pt model)
- Objects are detected using deep learning inference
- Bounding boxes and class labels are plotted
- Annotated frames are encoded into JPEG format

### 4. Real-Time Streaming

- Frames are streamed using multipart HTTP response
- Flask routes dynamically serve processed frames
- Users can switch between visualization modes

---

## Technologies Used

- Python
- Flask
- OpenCV
- NumPy
- Ultralytics YOLOv8
- HTML (Frontend rendering)

---

## Key Contributions

- Developed a multi-mode visual enhancement engine
- Integrated classical computer vision with deep learning
- Implemented Dark Channel Prior–based dehazing
- Built a real-time frame streaming architecture using Flask
- Designed a modular and scalable video analytics pipeline

---

## Practical Significance

This framework demonstrates how:

- Sensor simulation can be achieved using image processing
- Object detection performance can be analyzed under different visual conditions
- Real-time AI systems can be deployed using lightweight web frameworks
- Classical vision algorithms can complement deep learning models

---

## Future Improvements

- GPU acceleration for faster inference
- Integration of real-time webcam streaming
- Multi-object tracking support
- Deployment on cloud platforms
- Model scaling using YOLOv8n, YOLOv8m, and YOLOv8l variants
- Performance optimization using parallel processing

---

## How to Run

1. Clone the repository.
2. Install required dependencies using pip install -r requirements.txt.
3. Ensure yolov8s.pt model is available in the project directory.
4. Run the application using python app.py.
5. Open http://127.0.0.1:5000/ in your browser.
6. Upload a video and select the desired visualization mode.

---

## Author

Dhanush P  
Integrated M.Tech CSE (Business Analytics)  
Machine Learning and Computer Vision Project