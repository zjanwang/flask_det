from paddleocr import PaddleOCR
import cv2


class TextRecognizer:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True)

    def recognize_text(self, image_cv):
        # 使用 PaddleOCR 进行文字识别
        result = self.ocr.ocr(image_cv, cls=True)

        # 提取识别到的文字信息,并添加矩形框
        text_info = []
        for line in result:
            for word in line:
                bbox = word[0]
                text = word[1][0]
                left_top = (int(bbox[0][0]), int(bbox[0][1]))
                right_bottom = (int(bbox[2][0]), int(bbox[2][1]))
                cv2.rectangle(image_cv, left_top, right_bottom, (0, 255, 0), 2)
                cv2.putText(image_cv, text, left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                text_info.append(text)

        return text_info, image_cv
