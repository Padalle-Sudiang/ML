from flask import Flask, request
import cv2
import numpy as np
import os
import datetime
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)

model = YOLO("model.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def perform_ocr(image_array):
    results = ocr.ocr(image_array, rec=True)
    if results and results[0]:
        return ' '.join([res[1][0] for res in results[0]])
    return ""


def process_image_and_save(image, original_filename):
    os.makedirs("predict", exist_ok=True)
    image = cv2.resize(image, (1020, 500))
    results = model(image)
    annotator = Annotator(image, line_width=2)

    final_plate_text = ""
    final_plate_type = "Plat Biasa"  # Atau bisa dideteksi berdasarkan label model jika ada

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                cropped = image[y1:y2, x1:x2]
                ocr_text = perform_ocr(cropped)

                if ocr_text:
                    final_plate_text = ocr_text  # Ambil hasil OCR pertama sebagai plate_number
                    final_plate_type = class_name  # Misalnya label-nya 'Plat Biasa', 'Plat Dinas', dll

                label = f"{class_name}: {ocr_text}" if ocr_text else class_name
                annotator.box_label([x1, y1, x2, y2], label, colors(class_id, True))

    # Simpan gambar yang sudah dianotasi
    output_path = os.path.join("predict", f"result_{original_filename}")
    cv2.imwrite(output_path, image)
    print(f"[+] Prediction saved to: {output_path}")

    # Kirim ke API jika ada hasil OCR
    if final_plate_text:
        send_result_to_api(final_plate_text, final_plate_type, f"result_{original_filename}")



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


API_URL = "http://tkj-3b.com/tkj-3b.com/opengate/predict-plate.php"

def send_result_to_api(plate_number, plate_type, image_filename):
    API_URL = "http://tkj-3b.com/tkj-3b.com/opengate/predict-plate.php"
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
