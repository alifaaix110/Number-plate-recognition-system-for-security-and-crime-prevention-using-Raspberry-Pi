# main.py - ANPR SYSTEM (YOLO + OCR + Raspberry Pi)

import cv2
import numpy as np
import time
import re
import threading
from queue import Queue
from ultralytics import YOLO
from picamera2 import Picamera2
import pytesseract
import cvzone

# ---------------- CAMERA SETUP ---------------- #
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

# ---------------- LOAD MODEL ---------------- #
model = YOLO("models/best.pt")

# ---------------- LOAD PLATES ---------------- #
with open("config/plates.txt", "r") as f:
    plate_set = set(line.strip().upper() for line in f)

# ---------------- OCR ---------------- #
def ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# ---------------- THREADING ---------------- #
frame_queue = Queue(maxsize=1)

def capture():
    while True:
        frame = picam2.capture_array()
        if not frame_queue.full():
            frame_queue.put(frame)

threading.Thread(target=capture, daemon=True).start()

# ---------------- MAIN LOOP ---------------- #
history = {}

while True:

    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    frame = np.ascontiguousarray(frame)

    start = time.time()

    results = model.predict(frame, imgsz=192, conf=0.35, verbose=False)
    boxes = results[0].boxes.data

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            crop = frame[y1:y2, x1:x2]

            text = ocr(crop)

            if len(text) >= 4:
                if text in plate_set:
                    color = (0, 255, 0)
                    label = f"{text} MATCH"
                else:
                    color = (0, 0, 255)
                    label = f"{text} NO MATCH"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, label, (x1, y1 - 10), 1, 1, colorR=color)

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("ANPR System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()