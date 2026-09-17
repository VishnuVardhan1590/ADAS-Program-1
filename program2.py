# PROGRAM 2: Parking assistance / object detection using YOLO
import cv2
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
image_path = BASE / "inputs" / "parking.jpg"
output_file = BASE / "outputs" / "program2_parking_assistance.jpg"

model = YOLO("yolov8n.pt")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Input image not found: {image_path}")

results = model(image, verbose=False)
output = image.copy()

print("\n========== OBJECT DETECTION RESULTS ==========\n")
count = 0
for result in results:
    for box in result.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]
        width, height = x2-x1, y2-y1
        area = width*height

        if area > 120000: warning = "STOP!"
        elif area > 60000: warning = "BRAKE NOW"
        elif area > 25000: warning = "SLOW DOWN"
        else: warning = "SAFE"

        count += 1
        print("--------------------------------------")
        print("Class Name        :", label)
        print("Confidence        :", round(confidence,2))
        print("Bounding Box      :", (x1,y1,x2,y2))
        print("Bounding Box Area :", area)
        print("Estimated Status  :", warning)

        cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(output,f"{label} {confidence:.2f}",(x1,max(y1-30,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(output,warning,(x1,max(y1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

cv2.imwrite(str(output_file), output)
print(f"\nDetection completed. Objects detected: {count}")
print(f"Output saved: {output_file}")
cv2.imshow("Parking Assistance", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
