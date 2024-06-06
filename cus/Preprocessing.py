import base64


import cv2


class ImagePreprocessor:
    def __init__(self, scale_percent=50):
        self.scale_percent = scale_percent

    def preprocess_image(self, image_path,):
        # 读取图片并调整大小
        image = cv2.imread(image_path)

        width = int(image.shape[1] * self.scale_percent / 100)
        height = int(image.shape[0] * self.scale_percent / 100)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)



        # 将图片转换为 Base64 编码字符串
        retval, buffer = cv2.imencode('.jpg', resized_image)
        image_str = base64.b64encode(buffer).decode('utf-8')

        return image_str
