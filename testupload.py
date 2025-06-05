import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import mysql.connector
from paddleocr import PaddleOCR
import os
import glob


class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.db_connection = self.connect_to_db()

    def connect_to_db(self):
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password=""
            )
            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS vehicle_detection")
            print("Database 'vehicle_detection' checked/created.")

            connection.database = "vehicle_detection"
            create_table_query = """
            CREATE TABLE IF NOT EXISTS detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE,
                time TIME,
                class_name VARCHAR(255),
                numberplate TEXT,
                image_filename VARCHAR(255),
                confidence FLOAT
            )
            """
            cursor.execute(create_table_query)
            print("Table 'detections' checked/created.")

            return connection
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            raise

    def resize_keep_aspect(self, image, target_width=1020):
        h, w = image.shape[:2]
        aspect_ratio = h / w
        new_height = int(target_width * aspect_ratio)
        return cv2.resize(image, (target_width, new_height))

    def perform_ocr(self, image_array):
        if image_array is None or image_array.size == 0:
            return ""
        try:
            results = self.ocr.ocr(image_array, rec=True)
            if results and results[0]:
                return ' '.join([result[1][0] for result in results[0]])
            else:
                return ""
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def save_to_database(self, date, time, class_name, numberplate, image_filename, confidence):
        try:
            cursor = self.db_connection.cursor()
            query = """
                INSERT INTO detections (date, time, class_name, numberplate, image_filename, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (date, time, class_name, numberplate, image_filename, confidence))
            self.db_connection.commit()
            print(f"Data saved: {class_name} - {numberplate} - {image_filename}")
        except mysql.connector.Error as err:
            print(f"Error saving to database: {err}")

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return None

        image = self.resize_keep_aspect(image)

        results = self.model(image)
        annotator = Annotator(image, line_width=2)
        current_time = datetime.now()
        image_filename = os.path.basename(image_path)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    cropped_image = image[y1:y2, x1:x2]

                    ocr_text = self.perform_ocr(cropped_image)
                    label = f"{class_name}: {ocr_text}" if ocr_text.strip() else class_name

                    annotator.box_label([x1, y1, x2, y2], label=label, color=colors(class_id, True))

                    if ocr_text.strip():
                        self.save_to_database(
                            current_time.strftime("%Y-%m-%d"),
                            current_time.strftime("%H:%M:%S"),
                            class_name,
                            ocr_text,
                            image_filename,
                            confidence
                        )

        return image


def process_single_image(image_path, model_path="modelv2.pt"):
    detector = VehicleDetector(model_path)
    result = detector.process_image(image_path)

    if result is not None:
        cv2.imshow("Detection Result", result)
        output_path = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, result)
        print(f"Result saved as: {output_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def process_multiple_images(image_folder, model_path="best.pt"):
    detector = VehicleDetector(model_path)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))

    if not image_files:
        print(f"No image files found in {image_folder}")
        return

    print(f"Found {len(image_files)} images to process")
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        result = detector.process_image(image_path)
        if result is not None:
            output_filename = f"result_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result)
            print(f"Result saved as: {output_path}")


if __name__ == "__main__":
    single_image_path = "oo.jpeg"
    image_folder_path = "images/"
    model_path = "modelv2.pt"

    if os.path.exists(single_image_path):
        print("Processing single image...")
        process_single_image(single_image_path, model_path)
    elif os.path.exists(image_folder_path):
        print("Processing multiple images...")
        process_multiple_images(image_folder_path, model_path)
    else:
        print("Please specify a valid image path or folder path")
