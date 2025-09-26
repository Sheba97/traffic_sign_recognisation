import cv2
import numpy as np
from keras.models import load_model
from ImagePreprocessor import ImagePreprocessor  # your preprocessing class

# ------------------------------
# Load trained CNN model
# ------------------------------
model = load_model("traffic_sign_cnn_32.h5")  # replace with your trained Keras model path

# ------------------------------
# Color-based detection function
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

    # Combine all masks
    mask = red_mask | blue_mask | yellow_mask | white_mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # filter out small noise
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.7 < aspect_ratio < 1.3:  # roughly square
                detections.append((x, y, w, h))
    return detections


# ------------------------------
# Run detection + classification
# ------------------------------
def run_detection(image_path, labels):
    frame = cv2.imread(image_path)
    detections = detect_traffic_signs(frame)

    for (x, y, w, h) in detections:
        roi = frame[y:y+h, x:x+w]  # crop region of interest

        # Preprocess like training
        roi = cv2.resize(roi, (32, 32))
        roi = ImagePreprocessor.preprocess(roi)
        roi = roi.reshape(1, 32, 32, 1)

        # Predict
        pred = model.predict(roi)
        class_id = np.argmax(pred)
        prob = np.max(pred)

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{labels[class_id]} ({prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Traffic Sign Detection + Classification", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example labels for 9 classes (0-8) - replace with your actual labels.csv mapping
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

    image_path = "stop2.jpeg"  # 👈 put your test photo here
    run_detection(image_path, labels)