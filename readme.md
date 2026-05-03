# Automatic Number Plate Recognition (ANPR) System for crime prevention using Raspberry Pi.

## 📌 Project Overview
This project is an end-to-end real-time Automatic Number Plate Recognition (ANPR) system. It leverages the power of **YOLOv10** for high-speed object detection and **PyTesseract** for Optical Character Recognition (OCR). The system is trained on a custom dataset via Google Colab and deployed on a **Raspberry Pi 4** for real-time edge inference. 

Detected plates are matched against a local database to flag specific categories such as `CRIME`, `SIGNAL_BREAK`, or `PARKING`, and all detections are logged with timestamps.

---

## 🛠️ Hardware Requirements
- **Raspberry Pi 4 Model B**
- **Raspberry Pi Camera Module V2**
- **MicroSD Card** with **Raspberry Pi OS (Bookworm 64-bit)** installed

---

## 📦 Software & Dependencies
Ensure the following Python libraries are installed on your Raspberry Pi:
- `ultralytics` (YOLO)
- `opencv-python` (cv2)
- `picamera2` (Official Raspberry Pi camera library)
- `cvzone`
- `pytesseract` (Tesseract-OCR engine)
- `numpy`

*Note: You will also need to install the system-level Tesseract engine: `sudo apt-get install tesseract-ocr`*

---

## 🗂️ Dataset Preparation & Structure
1. **Data Collection:** Images were manually collected and supplemented with datasets from **Roboflow** and **Kaggle**.
2. **Annotation:** Tools like `labelImg` were used to draw bounding boxes and generate annotations in YOLO format (txt).
3. **Classes defined:**
   - `0`: `number-plate`
   - `1`: `character`
4. **Directory Structure:** The dataset was organized and zipped for training:
   ```text
   dataset/
   ├── images/
   │   ├── training/
   │   └── validation/
   └── labels/
       ├── training/
       └── validation/
   ```

---

## 🚀 Model Training (Google Colab)
The model was trained in the cloud using Google Colab (`rpi_cam_custom_dataset.ipynb`).

1. **Install Ultralytics:**
   ```bash
   !pip install ultralytics
   ```
2. **Mount Google Drive & Extract Dataset:** The zipped dataset was uploaded to Google Drive and extracted into the Colab environment.
3. **YOLOv10n Training:** The model was trained for 100 epochs using the custom `data.yaml` configuration.
   ```bash
   !yolo task=detect mode=train model=yolov10n.pt data=/content/datasets/freedomtech/data.yaml epochs=100 imgsz=256 plots=True
   ```
4. **Export to TFLite:** To ensure fast inference on the Raspberry Pi's CPU, the PyTorch model (`best.pt`) was exported to **TensorFlow Lite (FP32)** format.
   ```python
   from ultralytics import YOLO
   model = YOLO('/content/runs/detect/train3/weights/best.pt')
   model.export(format='tflite')
   ```

---

## 📂 Project Files Description
When deploying to the Raspberry Pi, ensure the following files are in your project directory:

- **`best_float32.tflite`** 
  The optimized YOLOv10 model exported from Colab.
- **`plates.txt`**
  Your local database mapping plate numbers to categories. Must be comma-separated: `[PLATE],[CATEGORY]`.
  *Example:*
  ```text
  OW07KLO,CRIME
  SX07GDK,SIGNAL_BREAK
  SKI8PHZ,PARKING
  BP65TXF,CRIME
  ```
- **`log.txt`**
  The system automatically creates/appends to this file. It records every recognized plate, its category, and the exact timestamp.
  *Example:*
  ```text
  2026-04-14 23:34:38.010342 | SX07GDK | SIGNAL_BREAK
  2026-04-14 23:53:39.868103 | SKI8PHZ | PARKING
  ```
- **`main.py`** *(or your script name)*
  The main inference script provided in the project.

---

## 🖥️ System Workflow & Logic
1. **Camera Feed Initialization:** `Picamera2` starts a threaded video stream at 320x240 resolution (RGB888 format).
2. **Detection:** The YOLO TFLite model scans frames for the `number-plate` class.
3. **Cropping & Pre-processing:** Detected plate bounding boxes are cropped, converted to grayscale, filtered, and adaptively thresholded to highlight text.
4. **Optical Character Recognition (OCR):** PyTesseract reads the pre-processed crop using specific configurations (`--psm 7` and alphanumeric whitelists) to extract text.
5. **Database Matching & UI Feedback:** The OCR text is checked against `plates.txt`. Bounding boxes are drawn on the live feed using specific colors based on the category:
   - 🔴 **CRIME** (Red)
   - 🟡 **SIGNAL_BREAK** (Yellow)
   - 🔵 **PARKING** (Blue)
   - 🟢 **MATCH** (Green - default for unflagged known plates)
6. **Logging:** Valid detections are logged to `log.txt`. A cooldown timer (`display_time`) prevents spam-logging the same plate multiple times per second.

---

## ▶️ Running the Application
To start the ANPR system on your Raspberry Pi:
1. Open your terminal.
2. Navigate to the project directory.
3. Run the python script:
   ```bash
   python main.py
   ```
*(Press the **'q'** key on your keyboard to safely exit the video window and stop the camera).*

## 👨‍💻 Author
Ali Faaiz
