import base64

import cv2
import numpy as np
import requests


def process_data(data):
    # 解码 Base64 编码的图片数据
    image_data = base64.b64decode(data['image'])

    # 将图片数据转换为 Numpy 数组
    buffer = np.frombuffer(image_data, np.uint8)

    # 解码图片数据为 OpenCV 图像格式
    image_cv = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image_cv


class Client:
    def __init__(self, server_url):
        self.server_url = server_url

    # 文字识别
    def recognize_text(self, image_str):
        # 发送 POST 请求到服务器
        response = requests.post(f'{self.server_url}/api/recognize_text', json={'image': image_str})

        # 解析服务器返回的 JSON 数据
        data = response.json()
        image_cv = process_data(data)
        # 显示图像
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('test', 800, 800)
        cv2.imshow('test', image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 打印识别到的文字信息
        print("识别结果:", data['text'])
    # 目标检测
    def detect_obj(self, image_str):
        # 发送 POST 请求到服务器
        response = requests.post(f'{self.server_url}/detect_objects', json={'image': image_str})

        # 解析服务器返回的 JSON 数据
        data = response.json()
        image_cv = process_data(data)

        # 显示图像
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('test', 800, 800)
        cv2.imshow('test', image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 圆检测
    def circle_det(self, image_str):
        # 发送 POST 请求到服务器
        response = requests.post(f'{self.server_url}/circle_det', json={'image': image_str})
        # 解析服务器返回的 JSON 数据
        data = response.json()
        image_cv = process_data(data)

        # 显示图像
        cv2.namedWindow('circle_distance', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('circle_distance', 800, 800)
        cv2.imshow('circle_distance', image_cv)
        print("distance", data['distance'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
