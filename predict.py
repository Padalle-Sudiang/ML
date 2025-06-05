from flask import Flask, request
import cv2
import numpy as np
import os
import datetime
import requests
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)

model = YOLO("modelv2.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

API_URL = "http://tkj-3b.com/tkj-3b.com/opengate/predict-plate.php"


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)
    return cv2.resize(image, dim, interpolation=inter)


def format_plate_text(text):
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', clean_text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return text 


def perform_ocr(image_array):
    results = ocr.ocr(image_array, rec=True)
    if results and results[0]:
        lines = [res[1][0] for res in results[0]]

        if lines:
            raw_text = lines[0] 
            return format_plate_text(raw_text)

    return ""


def process_image_and_save(image, original_filename):
    os.makedirs("predict", exist_ok=True)

    results = model(image)
    annotator = Annotator(image, line_width=2)

    final_plate_text = ""
    final_plate_type = None

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id] if class_id in model.names else "Unknown"
                cropped = image[y1:y2, x1:x2]
                ocr_text = perform_ocr(cropped)

                if ocr_text:
                    final_plate_text = ocr_text
                    final_plate_type = class_name

                label = f"{class_name}: {ocr_text}" if ocr_text else class_name
                annotator.box_label([x1, y1, x2, y2], label, colors(class_id, True))

    output_path = os.path.join("predict", f"result_{original_filename}")
    cv2.imwrite(output_path, image)
    print(f"[+] Prediction saved to: {output_path}")

    if final_plate_text and final_plate_type:
        send_result_to_api(final_plate_text, final_plate_type, f"result_{original_filename}")


def send_result_to_api(plate_number, plate_type, image_filename):
    image_path = os.path.join("predict", image_filename)

    data = {
        "plate_number": plate_number,
        "plate_type": plate_type,
    }

    try:
        with open(image_path, "rb") as img:
            files = {
                "image": img
            }
            response = requests.post(API_URL, data=data, files=files)
            print("[+] API Response:", response.text)
    except Exception as e:
        print("[!] Failed to send data to API:", e)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image provided", 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    filename = datetime.datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    process_image_and_save(img, filename)

    return f"Image processed and saved as {filename}", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
