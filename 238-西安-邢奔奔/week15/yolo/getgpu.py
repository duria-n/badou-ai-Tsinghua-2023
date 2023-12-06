# import tensorflow as tf
#
# # 创建 TensorFlow 会话
# with tf.compat.v1.Session() as sess:
#     # 获取可见的 GPU 设备列表
#     gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#
#     if gpu_devices:
#         # 打印 GPU 设备信息
#         for gpu in gpu_devices:
#             print(f"GPU Name: {gpu.name}")
#             print(f"GPU Device ID: {gpu.device_name}")
#     else:
#         print("No GPU devices found."

from PIL import Image

#
# with open('./model_data/yolov3.weights','rt') as fp:
#     dataall = np.fromfile(fp,dtype=np.float32)
#     print(dataall)
# num = np.array([[1,5,3],[3,4,7]])
# print(np.prod(num,axis=1))
from utils import letterbox_image

image_path = './img/img.jpg'
img = Image.open(image_path)
img_new = letterbox_image(img,(128,128))
img_new.show()