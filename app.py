import base64
import os

import cv2
import numpy as np

from flask import Flask, request, jsonify

from func import circle_det
from func.text_rec import TextRecognizer
from func.obj_detect import Detect

# 初始化 Flask 应用
app = Flask(__name__)
path = 'D:\PycharmProjects\\flaskProject\Resources'


# 定义图片文字识别接口
@app.route('/api/recognize_text', methods=['POST'])
def recognize_text():
    # 接收前端发送的图片数据
    data = request.json
    image_data = data.get('image', None)
    image_cv = base64_to_opencv(image_data)

    # 初始化 TextRecognizer
    text_recognizer = TextRecognizer()

    # 调用 TextRecognizer 实例的 recognize_text 方法进行文字识别
    text_info, image_cv = text_recognizer.recognize_text(image_cv)
    retval, buffer = cv2.imencode('.jpg', image_cv)
    image_str = base64.b64encode(buffer).decode('utf-8')
    # 返回识别到的文字信息
    return jsonify({'image': image_str, 'text': text_info})

# 定义目标检测接口
@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    if request.method == "POST":
        # 接收前端发送的图片数据
        data = request.json
        image_data = data.get('image', None)
        image_cv = base64_to_opencv(image_data)
        # timestamp = datetime.now().strftime('%Y%m%d')  # 当前日期
        # filename = f"{timestamp}"
        if not os.path.exists(path):
            os.makedirs(path)
        filename = 't2.jpg'
        save_path = f'{path}\\{filename}'
        cv2.imwrite(f'{save_path}', image_cv)

        detector = Detect()
        # 使用Detect类进行物体检测
        image_cv = detector.detect_objects(save_path, filename)

        retval, buffer = cv2.imencode('.jpg', image_cv)
        # 将结果图像转换为base64编码发送给客户端
        image_str = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': image_str})
    else:
        return jsonify({"error": "Method not allowed"}), 405

# 定义圆检测接口
@app.route("/circle_det", methods=["POST"])
def circle_det():
    if request.method == "POST":
        data = request.json
        image_data = data.get('image', None)
        image_cv = base64_to_opencv(image_data)
        distance, image_cv = circle_det.detect_and_draw_line(image_cv)
        _, buffer = cv2.imencode('.jpg', image_cv)
        image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'distance': distance, 'image': image})


def base64_to_opencv(image_str):
    try:
        # 解码 Base64 字符串为原始字节数据
        image_data = base64.b64decode(image_str)

        # 将原始字节数据转换为 NumPy 数组
        nparr = np.frombuffer(image_data, np.uint8)

        # 使用 OpenCV 解码图像数组
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return image_cv
    except Exception as e:
        print("Error:", e)
        return None


# 启动应用
if __name__ == '__main__':
    app.run(debug=True)
