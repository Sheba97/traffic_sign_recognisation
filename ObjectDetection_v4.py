import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from ImagePreprocessor import ImagePreprocessor  # your preprocessing class

# ------------------------------
# Load trained CNN model
# ------------------------------
model = load_model("traffic_sign_cnn_32.h5")  # replace with your trained model
if model is None:
    raise RuntimeError("Failed to load the Keras model. Please check the model path.")

# ------------------------------
# Define labels (replace with your actual dataset mapping)
# ------------------------------
labels = {
    0: "Speed Limit 50",
    1: "Speed Limit 100",
    2: "Stop",
    3: "Construction",
    4: "Traffic signals",
    5: "Turn right ahead",
    6: "Turn left ahead",
    7: "Intersection",
    8: "Tunnel",
    9: "Parking"
}

# ------------------------------
# Color + shape detection
# ------------------------------
def detect_traffic_signs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- RED ---
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # --- BLUE ---
    blue_lower = np.array([90, 70, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # --- YELLOW ---
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # --- WHITE ---
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # Combine masks
    mask = red_mask | blue_mask | yellow_mask | white_mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # Approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)

        x, y, w, h = cv2.boundingRect(cnt)

        shape = "Unknown"
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            aspect_ratio = w / float(h)
            shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif 6 <= vertices <= 8:
            shape = "Diamond"
        else:
            circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2)
            if circularity > 0.7:
                shape = "Circle"

        detections.append((x, y, w, h, shape))
    return detections

# ------------------------------
# Run detection + classification for a single image
# ------------------------------
def run_on_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    detections = detect_traffic_signs(frame)

    for (x, y, w, h, shape) in detections:
        roi = frame[y:y + h, x:x + w]

        try:
            roi = cv2.resize(roi, (32, 32))
            roi = ImagePreprocessor.preprocess(roi)
            roi = roi.reshape(1, 32, 32, 1)

            # Predict
            pred = model.predict(roi, verbose=0)
            class_id = np.argmax(pred)
            prob = np.max(pred)

            # Draw results
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{labels.get(class_id, 'Unknown')} ({prob:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Shape: {shape}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e:
            print(f"ROI processing error: {e}")
            continue

    cv2.imshow("Traffic Sign Detection + Classification", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# Run test
# ------------------------------
if __name__ == "__main__":
    target_size = (32, 32)
    test_image = "12_cycles.png"  # replace with your test image
    run_on_image(test_image)
