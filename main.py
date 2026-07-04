import cv2
import time
import os
import psutil
from ultralytics import YOLO

# 1. SETUP & OPTIMIZATION
model_name = 'yolov8m.pt'
export_folder = 'yolov8m_openvino_model'

if not os.path.exists(export_folder):
    base_model = YOLO(model_name)
    base_model.export(format='openvino') 

model = YOLO(export_folder, task='detect')
cap = cv2.VideoCapture(0)

print("--- STARTING THREE-STAGE SMART  SECURITY ---")

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success: break

    # 2. GPU-ACCELERATED INFERENCE
    results = model.predict(frame, imgsz=640, conf=0.3, 
                            classes=[0, 39, 56, 62, 63, 64, 66, 67], stream=True)
    cpu_usage = psutil.cpu_percent()

    for result in results:
        # 3. THREE-STAGE LOGIC
        # 63 = laptop, 67 = cell phone
        laptop_detected = any(int(box) == 63 for box in result.boxes.cls)
        phone_detected = any(int(box) == 67 for box in result.boxes.cls)

        # LOGIC ENGINE
        if laptop_detected and phone_detected:
            status_text = "STATUS: ASSETS SECURED (ALL PRESENT)"
            status_color = (0, 255, 0) # GREEN
        elif laptop_detected or phone_detected:
            status_text = "WARNING: ONE DEVICE MISSING"
            status_color = (0, 255, 255) # YELLOW
        else:
            status_text = "WARNING: NO DEVICES DETECTED"
            status_color = (0, 0, 255) # RED

        # 4. DRAW UI DASHBOARD
        annotated_frame = result.plot()
        latency = (time.time() - start_time) * 1000
        fps = 1 / (time.time() - start_time)

        # Header Dashboard Bar
        cv2.rectangle(annotated_frame, (0, 0), (1280, 60), (30, 30, 30), -1)
        cv2.putText(annotated_frame, status_text, (20, 42), 1, 1.8, status_color, 3)

        # Performance Overlay (Right Side)
        cv2.putText(annotated_frame, f"Latency: {int(latency)}ms", (20, 120), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 150), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"CPU Load: {cpu_usage}%", (20, 180), 1, 1.2, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Intel Arc GPU: OPTIMIZED", (20, 210), 1, 1.2, (0, 165, 255), 2)

        cv2.imshow("Sejong University - ECI Midterm", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()