# PROGRAM 3: Vehicle distance + Time To Collision (TTC) from video
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
video_path = BASE / "inputs" / "test.mp4"
output_path = BASE / "outputs" / "program3_ttc_output.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 10
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))

FOCAL_LENGTH = 700
KNOWN_WIDTH = 1.8
previous_distance = None
previous_time = None
smoothed_ttc = None

def estimate_distance(box_width):
    return float("inf") if box_width <= 0 else (KNOWN_WIDTH*FOCAL_LENGTH)/box_width

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_no += 1
    current_time = time.time()
    results = model(frame, verbose=False)[0]

    best_distance = None
    best_box = None

    for box in results.boxes:
        name = model.names[int(box.cls[0])]
        if name not in ["car","truck","bus","motorcycle"]:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        distance = estimate_distance(x2-x1)
        if best_distance is None or distance < best_distance:
            best_distance, best_box = distance, (x1,y1,x2,y2)

    displayed_ttc = float("inf")
    if best_distance is not None:
        if previous_distance is not None and previous_time is not None:
            dt = max(current_time-previous_time, 1e-3)
            relative_speed = (previous_distance-best_distance)/dt
            if relative_speed > 0:
                ttc = best_distance/relative_speed
                smoothed_ttc = ttc if smoothed_ttc is None else 0.7*smoothed_ttc+0.3*ttc
        previous_distance, previous_time = best_distance, current_time
        displayed_ttc = smoothed_ttc if smoothed_ttc is not None else float("inf")

        x1,y1,x2,y2 = best_box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"Distance: {best_distance:.1f} m",(x1,max(y1-10,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    if displayed_ttc < 2: warning = "CRITICAL: COLLISION WARNING!"
    elif displayed_ttc < 5: warning = "CAUTION: REDUCE SPEED"
    else: warning = "SAFE"

    cv2.putText(frame,warning,(25,45),cv2.FONT_HERSHEY_SIMPLEX,0.8,
                (0,0,255) if "CRITICAL" in warning else
                (0,165,255) if "CAUTION" in warning else (0,255,0),3)
    cv2.putText(frame,f"TTC: {displayed_ttc:.1f} s",(25,85),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    writer.write(frame)
    cv2.imshow("TTC / Collision Warning", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Processed {frame_no} frames.")
print(f"Output saved: {output_path}")
