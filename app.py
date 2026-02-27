

from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

# Load YOLOv8 model once
model = YOLO("yolov8s.pt")

# Global variable to hold uploaded video temporarily
uploaded_video_path = None

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_video_path
    if request.method == "POST":
        file = request.files["video"]
        if file:
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file.save(temp_file.name)
            uploaded_video_path = temp_file.name
            return render_template("index.html", show_video=True)
    return render_template("index.html", show_video=False)

def apply_lidar_effect(frame):
    """Apply LIDAR-like visualization effect with reduced graininess."""
    # Convert to grayscale and apply bilateral filter to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Create a colormap similar to LIDAR visualization
    height, width = gray.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create depth-like effect with smoother transitions
    colored[gray > 30] = [0, 0, 255]    # Far objects in blue
    colored[gray > 90] = [128, 0, 255]  # Mid-far objects in purple
    colored[gray > 150] = [255, 0, 255] # Mid-range objects in magenta
    colored[gray > 200] = [255, 165, 0] # Near objects in orange
    
    # Add point cloud effect with reduced graininess
    mask = np.random.random((height, width)) > 0.5  # Reduced sparsity
    colored[~mask] = [0, 0, 0]
    
    # Apply slight blur to smooth out the visualization
    colored = cv2.GaussianBlur(colored, (3, 3), 0)
    
    return colored

def apply_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # Enhance contrast
    thermal = cv2.convertScaleAbs(thermal, alpha=1.2, beta=10)
    return thermal

def get_dark_channel(image, patch_size=15):
    """Get dark channel prior for dehazing."""
    b, g, r = cv2.split(image)
    min_rgb = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_rgb, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """Estimate atmospheric light in the image."""
    h, w = dark_channel.shape
    flat_image = image.reshape(h * w, 3)
    flat_dark = dark_channel.ravel()
    
    # Pick the top 0.1% brightest pixels in the dark channel
    top_pixels = int(h * w * 0.001)
    bright_locations = np.argpartition(flat_dark, -top_pixels)[-top_pixels:]
    
    # Return the highest intensity in the input image
    return np.max(flat_image[bright_locations], axis=0)

def estimate_transmission(image, A, patch_size=15, omega=0.95):
    """Estimate transmission map."""
    normalized = image / A
    transmission = 1 - omega * get_dark_channel(normalized, patch_size)
    return transmission

def guided_filter(guide, transmission, radius=50, eps=0.001):
    """Apply guided filter to refine transmission map."""
    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_transmission = cv2.boxFilter(transmission, -1, (radius, radius))
    mean_gt = cv2.boxFilter(guide * transmission, -1, (radius, radius))
    cov_gt = mean_gt - mean_guide * mean_transmission

    mean_guide_square = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_guide = mean_guide_square - mean_guide * mean_guide

    a = cov_gt / (var_guide + eps)
    b = mean_transmission - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    return mean_a * guide + mean_b

def apply_dehaze(image, patch_size=15, omega=0.95, t0=0.1, radius=50, eps=0.001):
    """Apply dehazing to an image."""
    # Convert to float for calculations
    image_float = image.astype(np.float32) / 255
    
    # Get dark channel prior
    dark_channel = get_dark_channel(image_float, patch_size)
    
    # Estimate atmospheric light
    A = estimate_atmospheric_light(image_float, dark_channel)
    
    # Estimate transmission map
    transmission = estimate_transmission(image_float, A, patch_size, omega)
    
    # Refine transmission map using guided filter
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    transmission = guided_filter(gray_image, transmission, radius, eps)
    
    # Ensure minimum transmission
    transmission = np.maximum(transmission, t0)
    
    # Recover dehazed image
    result = np.empty_like(image_float)
    for i in range(3):
        result[:, :, i] = (image_float[:, :, i] - A[i]) / transmission + A[i]
    
    # Clip values and convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    # Enhance contrast and brightness
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return result

def gen_frames(effect_func):
    if not uploaded_video_path:
        return
        
    cap = cv2.VideoCapture(uploaded_video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        try:
            processed = effect_func(frame)
            results = model(processed, verbose=False)
            for result in results:
                annotated = result.plot()
                _, buffer = cv2.imencode(".jpg", annotated)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
            
    cap.release()

@app.route("/video_feed_lidar")
def video_feed_lidar():
    return Response(gen_frames(apply_lidar_effect),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_thermal")
def video_feed_thermal():
    return Response(gen_frames(apply_thermal),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_dehaze")
def video_feed_dehaze():
    return Response(gen_frames(apply_dehaze),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)

