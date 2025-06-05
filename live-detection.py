import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
import mysql.connector
from paddleocr import PaddleOCR
import os
import time


class LiveVehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        self.db_connection = self.connect_to_db()
        
        self.saved_detections = set()
        
        self.screenshot_counter = 0
        
        os.makedirs("screenshots", exist_ok=True)

    def connect_to_db(self):
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password=""
            )
            cursor = connection.cursor()

            cursor.execute("CREATE DATABASE IF NOT EXISTS live_vehicle_detection")
            print("Database 'live_vehicle_detection' checked/created.")

            connection.database = "live_vehicle_detection"

            create_table_query = """
            CREATE TABLE IF NOT EXISTS live_detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE,
                time TIME,
                class_name VARCHAR(255),
                numberplate TEXT,
                screenshot_filename VARCHAR(255),
                confidence FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            print("Table 'live_detections' checked/created.")

            return connection
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            raise

    def perform_ocr(self, image_array):
        if image_array is None or image_array.size == 0:
            return ""
        
        if image_array.shape[0] < 20 or image_array.shape[1] < 20:
            return ""
        
        try:
            results = self.ocr.ocr(image_array, rec=True)
            if results and results[0]:
                text = ' '.join([result[1][0] for result in results[0]])
                cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip()
                return cleaned_text
            else:
                return ""
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def save_to_database(self, date, time_str, class_name, numberplate, screenshot_filename, confidence):
        try:
            cursor = self.db_connection.cursor()
            query = """
                INSERT INTO live_detections (date, time, class_name, numberplate, screenshot_filename, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (date, time_str, class_name, numberplate, screenshot_filename, confidence))
            self.db_connection.commit()
            print(f"✅ Saved: {class_name} - {numberplate} - {screenshot_filename}")
        except mysql.connector.Error as err:
            print(f"❌ Database error: {err}")

    def save_screenshot(self, frame, class_name, numberplate):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.screenshot_counter += 1
        filename = f"screenshot_{timestamp}_{self.screenshot_counter}_{class_name}_{numberplate.replace(' ', '_')}.jpg"
        filepath = os.path.join("screenshots", filename)
        
        cv2.imwrite(filepath, frame)
        return filename

    def process_frame(self, frame):
        results = self.model(frame)
        
        annotator = Annotator(frame, line_width=2)
        
        current_time = datetime.now()
        
        detections_found = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if confidence < 0.5:
                        continue
                    
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    ocr_text = self.perform_ocr(cropped_image)
                    
                    if ocr_text.strip():
                        label = f"{class_name}: {ocr_text} ({confidence:.2f})"
                        
                        detection_id = f"{class_name}_{ocr_text}_{current_time.strftime('%H%M')}"
                        
                        if detection_id not in self.saved_detections:
                            screenshot_filename = self.save_screenshot(frame, class_name, ocr_text)
                            
                            self.save_to_database(
                                current_time.strftime("%Y-%m-%d"),
                                current_time.strftime("%H:%M:%S"),
                                class_name,
                                ocr_text,
                                screenshot_filename,
                                confidence
                            )
                            
                            self.saved_detections.add(detection_id)
                            detections_found.append(f"{class_name}: {ocr_text}")
                    else:
                        label = f"{class_name} ({confidence:.2f})"
                    
                    annotator.box_label([x1, y1, x2, y2], label=label, color=colors(class_id, True))

        return frame, detections_found

    def run_live_detection(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Live detection started!")
        print("📝 Instructions:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to manually save screenshot")
        print("   - Press 'c' to clear saved detections cache")
        print("   - Detections with numberplates are automatically saved")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame from camera")
                break
            
            processed_frame, detections = self.process_frame(frame)
            
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
            else:
                fps = 0
            
            cv2.putText(processed_frame, f"Live Vehicle Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if fps > 0:
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(processed_frame, "Press 'q' to quit", (10, processed_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if detections:
                y_offset = 110
                for detection in detections[-5:]:
                    cv2.putText(processed_frame, f"✅ {detection}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
            
            cv2.imshow("Live Vehicle Detection", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"manual_screenshot_{timestamp}.jpg"
                cv2.imwrite(os.path.join("screenshots", filename), processed_frame)
                print(f"📸 Manual screenshot saved: {filename}")
            elif key == ord('c'):
                self.saved_detections.clear()
                print("🗑️  Detection cache cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🔚 Live detection stopped")

    def run_video_file_detection(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file {video_path}")
            return
        
        print(f"🎬 Processing video: {video_path}")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 3 != 0:
                continue
            
            frame = cv2.resize(frame, (1280, 720))
            
            processed_frame, detections = self.process_frame(frame)
            
            progress = (frame_count / total_frames) * 100
            cv2.putText(processed_frame, f"Progress: {progress:.1f}% ({frame_count}/{total_frames})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Video Detection", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
        print("🔚 Video processing completed")


def main():
    model_path = "modelv2.pt"
    
    print("🚗 Live Vehicle Detection System")
    print("=" * 40)
    print("Choose detection mode:")
    print("1. Live detection (Webcam)")
    print("2. Video file detection")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        detector = LiveVehicleDetector(model_path)
        camera_index = input("Enter camera index (default 0): ").strip()
        camera_index = int(camera_index) if camera_index.isdigit() else 0
        detector.run_live_detection(camera_index)
        
    elif choice == "2":
        video_path = input("Enter video file path: ").strip()
        if os.path.exists(video_path):
            detector = LiveVehicleDetector(model_path)
            detector.run_video_file_detection(video_path)
        else:
            print(f"❌ Video file not found: {video_path}")
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()