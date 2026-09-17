# PROGRAM 1: Image classification + ambient brightness/headlight decision
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

BASE = __import__("pathlib").Path(__file__).resolve().parent
image_path = BASE / "inputs" / "green.jpeg"

print("TensorFlow Version:", tf.__version__)
model = MobileNetV2(weights="imagenet")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Input image not found: {image_path}")

img = cv2.resize(image, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = preprocess_input(np.expand_dims(img.astype(np.float32), axis=0))
predictions = model.predict(x, verbose=0)
results = decode_predictions(predictions, top=3)[0]

print("\nTop 3 Predictions:")
for _, label, probability in results:
    print(f"{label}: {probability*100:.2f}%")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
brightness = float(np.mean(gray))
print(f"\nAmbient Brightness: {brightness:.2f}")
print("Headlights ON" if brightness < 80 else "Headlights OFF")

cv2.imwrite(str(BASE/"outputs"/"program1_input.jpg"), image)
print("Output saved: outputs/program1_input.jpg")
