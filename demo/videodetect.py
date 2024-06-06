import cv2
import numpy as np
from ultralytics import YOLO

# 视频异常检测
# 加载异常检测模型
model = YOLO('../model/yolo_pip.pt')

# 打开视频文件
video_capture = cv2.VideoCapture('../Resources/pip/pip_1.avi')  # 示例，假设视频文件为mp4格式

# 获取视频的帧率和分辨率
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 打开视频输出流


fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 输出文件保存位置
out = cv2.VideoWriter('../result/out/seg_pip_1.avi', fourcc, fps, (width, height))

# 逐帧处理视频
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # 在这里对每一帧图像进行异常检测
    detections = model.predict(frame)

    # 将检测结果绘制在原始帧上
    det_plotted = detections[0].plot()

    # 将 PIL 格式的图像转换为 OpenCV 格式
    frame_with_det = cv2.cvtColor(np.array(det_plotted), cv2.COLOR_RGB2BGR)

    # 将标注后的图像帧写入视频输出流
    out.write(frame_with_det)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象、视频输出流
video_capture.release()
out.release()
