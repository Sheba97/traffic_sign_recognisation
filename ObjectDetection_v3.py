import cv2
import numpy as np
from keras.models import load_model

# =====================
# Load your trained CNN
# =====================
cnn_model = load_model("traffic_sign_cnn_32.h5")

# Label map (adjust for your dataset)
labels = {
        0: "Speed Limit 30",
        1: "Speed Limit 50",
        2: "Stop",
        3: "Construction",
        4: "Traffic signals",
        5: "Turn right ahead",
        6: "Turn left ahead",
        7: "Intersection",
        8: "Tunnel",
        9: "Parking"
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
        roi_input = roi_norm.reshape(1, 32, 32, 1)  # (batch, H, W, C)

        pred = model.predict(roi_input, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        return class_id, confidence
    except Exception as e:
        print(f"Classification error: {e}")
        return None, None

# =====================
# Shape + Color Detection
# =====================
def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red, Blue, Yellow filters
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

        if w * h < 500:
            continue

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
# Video Detection Loop
# =====================
def run_video(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)  # webcam if no path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_shapes(frame)

        for (x, y, w, h, shape) in detections:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            class_id, conf = classify_with_cnn(roi, cnn_model)
            if class_id is None:
                continue

            label_text = f"{labels[class_id]} ({conf:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Traffic Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================
# Run Example
# =====================
if __name__ == "__main__":
    run_video("AutoraceTrackLab61All.mp4")  # change to None for webcam
