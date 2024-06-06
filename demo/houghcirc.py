import cv2
import numpy as np
# 该文件为视频检测圆

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
# 打开视频文件
video_capture = cv2.VideoCapture('../Resources/v7.avi')

# 获取视频的帧率和分辨率
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 打开视频输出流
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 输出视频保存位置
out = cv2.VideoWriter('../result/out/circle_det.avi', fourcc, fps, (width, height))


# 初始化上一帧的圆心列表
prev_centers = []
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行模糊处理，以减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 进行圆检测
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=200, param1=200, param2=30, minRadius=50,
                               maxRadius=300)

    # 确保至少检测到了一个圆
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")


        for (x, y, r) in circles:
            # 绘制圆和圆心
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), 4)

        # 将标记后的帧写入输出视频流
    out.write(frame)

# 释放视频捕获对象、视频输出流和关闭所有窗口
video_capture.release()
out.release()
cv2.destroyAllWindows()
# 显示图像
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 800, 800)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# else:
#     print("No circles detected.")
