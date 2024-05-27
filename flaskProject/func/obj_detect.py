import os
import cv2

from ultralytics import YOLO



class Detect:
    def __init__(self,
                 model_path=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "model", 'yolo_detect.pt'))):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            print("Error loading model:", e)
            return None

    def detect_objects(self, path, name):
        detections = self.model.predict(path)
        det_plotted = detections[0].plot()
        cv2.imwrite(f'D:\PycharmProjects\\flaskProject\\result\{name}', det_plotted)
        return det_plotted

# yolo detect predict model=model/yolo_detect.pt source=Resources/test.jpg
