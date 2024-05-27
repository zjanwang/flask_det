from cus.Preprocessing import ImagePreprocessor
from cus.Client import Client

# 测试图片文件路径
image_file_path = '../Resources/t1.jpg'

# 使用 ImagePreprocessor 预处理图片
imagePreprocessor = ImagePreprocessor()
image_str = imagePreprocessor.preprocess_image(image_file_path)

# 创建 OCRClient 实例并发送请求
client = Client(server_url='http://localhost:5000')

# 测试文字识别
# client.recognize_text(image_str)

# 测试目标检测
# client.detect_obj(image_str)

# 测试圆检测
client.circle_det(image_str)




