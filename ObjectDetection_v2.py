import cv2
import numpy as np
from keras.models import load_model

# =====================
# Load your trained CNN
# =====================
cnn_model = load_model("traffic_sign_cnn_32.h5")   # make sure this path is correct

# Label map (adjust for your dataset)
labels = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "No Entry",
    4: "Stop",
    5: "Yield",
    6: "Turn Right",
    7: "Turn Left",
    8: "Construction"
}

# =====================
# ROI Classification
# =====================
def classify_with_cnn(roi, model):
    try:
        roi_resized = cv2.resize(roi, (32, 32))
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        roi_eq = cv2.equalizeHist(roi_gray)
        roi_norm = roi_eq / 255.0
        roi_input = roi_norm.reshape(1, 32, 32, 1)  # shape (1, H, W, C)

        pred = model.predict(roi_input, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        return class_id, confidence
    except Exception as e:
        print(f"Error in classify_with_cnn: {e}")
        return None, None

# =====================
# Shape + Color Detection
# =====================
def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red, Blue, Yellow masks (common traffic sign colors)
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    blue_mask = cv2.inRange(hsv, (90, 60, 0), (121, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))

    mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), yellow_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # Filter small boxes
        if w * h < 500:
            continue

        # Determine shape type
        shape = "Unknown"
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) == 5:
            shape = "Pentagon"
        elif 6 < len(approx) < 15:
            shape = "Circle"
        elif len(approx) >= 15:
            shape = "Ellipse"

        detections.append((x, y, w, h, shape))

    return detections

# =====================
# Main Detection + Classification
# =====================
def run_detection(image_path):
    frame = cv2.imread(image_path)
    detections = detect_shapes(frame)

    for (x, y, w, h, shape) in detections:
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        class_id, conf = classify_with_cnn(roi, cnn_model)
        if class_id is None:
            continue

        label_text = f"{labels[int(class_id)]} ({conf:.2f})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =====================
# Run example
# =====================
if __name__ == "__main__":
    image_path = "stop.jpeg"  # replace with your test image
    run_detection(image_path)
