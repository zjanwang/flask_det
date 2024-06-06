import cv2
import numpy as np

# 定义全局变量用于存储鼠标点击的点坐标和绘制的直线
points = []
lines = []
circle_radius = 10  # 设置圆的半径

# 读取图像
image = cv2.imread('../Resources/test.jpg')  # 示例，假设图像文件为jpg格式

# 定义鼠标点击事件处理函数
def mouse_click(event, x, y, flags, param):
    global points, lines

    # 创建图像的副本，以避免直接修改原始图像
    img = image.copy()

    # 当左键按下时，记录鼠标点击的点坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # 在图像上绘制点击的点
        cv2.circle(img, (x, y), circle_radius, (0, 255, 0), -1)  # 绘制圆形标记点

        # 当点击的点数达到两个时，计算两点之间的距离，并绘制直线
        if len(points) == 2:
            distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            print("距离:", distance)

            # 在图像上绘制直线和距离信息
            cv2.line(img, points[0], points[1], (0, 0, 255), 2)  # 绘制直线
            cv2.putText(img, f"{distance:.2f}",
                        (int((points[0][0] + points[1][0]) / 2), int((points[0][1] + points[1][1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 显示距离信息

            # 保存绘制的直线
            lines.append((points[0], points[1]))

            # 清空点列表，准备接收下一次点击
            points = []

    # 绘制已保存的直线和文字
    for line in lines:
        cv2.line(img, line[0], line[1], (0, 0, 255), 10)
        distance = np.linalg.norm(np.array(line[0]) - np.array(line[1]))
        cv2.putText(img, f"{distance:.2f}",
                    (int((line[0][0] + line[1][0]) / 2), int((line[0][1] + line[1][1]) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)  # 显示距离信息

    # 显示更新后的图像
    cv2.imshow('image', img)


# 在图像上设置鼠标点击事件回调函数
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 800)
cv2.imshow('image', image)
cv2.setMouseCallback('image', mouse_click)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 释放窗口
cv2.destroyAllWindows()
