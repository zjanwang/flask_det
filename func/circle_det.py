import cv2
import numpy as np


def detect_and_draw_line(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测圆
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10,
                               maxRadius=100)

    if circles is not None:
        # 将圆心坐标转换为整数
        circles = np.round(circles[0, :]).astype("int")

        # 如果检测到了至少两个圆
        if len(circles) >= 2:
            # 计算两个圆心的距离
            center1 = circles[0][:2]
            center2 = circles[1][:2]
            distance = np.linalg.norm(center1 - center2)
            distance = distance*0.025
            # 计算直线的中点
            line_center = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)

            # 绘制直线
            cv2.circle(image, tuple(center1), 0, (0, 255, 0), 4)
            cv2.circle(image, tuple(center2), 0, (0, 255, 0), 4)
            cv2.line(image, tuple(center1), tuple(center2), (0, 0, 0), 3)

            # 在直线上方绘制距离
            cv2.putText(image, f"{distance:.2f}cm", (line_center[0], line_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            # 保存绘制直线后的图像
            cv2.imwrite("result_image.jpg", image)

            return distance, image
        else:
            return "Not enough circles detected"
    else:
        return "No circles detected"


