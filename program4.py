# PROGRAM 4: Histogram equalization
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
img = cv2.imread(str(BASE/"inputs"/"histogram_input.png"))
if img is None:
    raise FileNotFoundError("histogram_input.png not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
he_img = cv2.equalizeHist(gray)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(gray,cmap="gray"); plt.axis("off")
plt.subplot(1,3,2); plt.title("Histogram Equalized"); plt.imshow(he_img,cmap="gray"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Histogram"); plt.hist(he_img.ravel(), bins=256, range=(0,256))
plt.tight_layout()
plt.savefig(str(BASE/"outputs"/"program4_histogram_equalization.png"), dpi=150)
plt.show()
print("Output saved: outputs/program4_histogram_equalization.png")
