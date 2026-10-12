import cv2

image = cv2.imread("road.jpg")

if image is None:
    print("Image is not found")
    exit()

hog = cv2.HOGDescriptor()

hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

rects, weights = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.03
)

for (x, y, w, h), weight in zip(rects, weights):
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Person {float(weight):.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

print("Number of pedestrians:", len(rects))

cv2.imshow("HOG Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()