import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
import re
import time
import threading
from queue import Queue
from datetime import datetime

# ---------------- CAMERA SETUP ---------------- #
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)

# ---------------- LOAD YOLO MODEL ---------------- #
model = YOLO('best.pt')
device_type = "cpu"

# ---------------- LOAD PLATES WITH CATEGORY ---------------- #
plate_dict = {}

with open("plates.txt", "r") as f:
    for line in f:
        parts = line.strip().upper().replace(" ", "").split(",")
        if len(parts) == 2:
            plate_dict[parts[0]] = parts[1]

print("Loaded Plates:", plate_dict)

# ---------------- LOG FILE ---------------- #
def log_plate(plate, category):
    with open("log.txt", "a") as log:
        log.write(f"{datetime.now()} | {plate} | {category}\n")

# ---------------- OCR FUNCTION ---------------- #
def perform_ocr(img):
    if img is None or img.size == 0:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.resize(gray, (220, 90))

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    text = pytesseract.image_to_string(
        thresh,
        config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    return re.sub(r'[^A-Z0-9]', '', text.upper()).strip()

# ---------------- THREAD ---------------- #
frame_queue = Queue(maxsize=1)

def capture_frames():
    while True:
        frame = picam2.capture_array()
        if not frame_queue.full():
            frame_queue.put(frame)

threading.Thread(target=capture_frames, daemon=True).start()

# ---------------- VARIABLES ---------------- #
detected_history = {}
display_time = 2

# ---------------- MAIN LOOP ---------------- #
while True:

    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    frame = np.ascontiguousarray(frame)
    h, w, _ = frame.shape

    start_time = time.time()

    results = model.predict(frame, imgsz=192, conf=0.35, device=device_type, verbose=False)
    boxes = results[0].boxes.data

    if boxes is not None and len(boxes) > 0:

        for box in boxes:

            x1, y1, x2, y2 = map(int, box[:4])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) < 50 or (y2 - y1) < 25:
                continue

            crop = frame[y1:y2, x1:x2]

            text = perform_ocr(crop)

            if len(text) >= 4:

                current_time = time.time()

                if text not in detected_history or (current_time - detected_history[text]) > display_time:
                    detected_history[text] = current_time

                    # ---------------- NEW PART ONLY ---------------- #
                    if text in plate_dict:
                        category = plate_dict[text]

                        if category == "CRIME":
                            color = (0, 0, 255)
                        elif category == "SIGNAL_BREAK":
                            color = (0, 255, 255)
                        elif category == "PARKING":
                            color = (255, 0, 0)
                        else:
                            color = (0, 255, 0)

                        label = f"{text} {category}"
                        print(f"{category}: {text}")

                        log_plate(text, category)

                    else:
                        color = (0, 255, 0)
                        label = f"{text} MATCH"

                    # ---------------- DRAW ---------------- #
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cvzone.putTextRect(frame, label, (x1, max(20, y1 - 10)), 1, 1, colorR=color)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # ---------------- FPS ---------------- #
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("ANPR Fast Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()