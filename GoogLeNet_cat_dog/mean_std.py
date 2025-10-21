from PIL import Image
import os
import numpy as np

# 文件夹路径，包含所有图片文件
folder_path = 'data_cat_dog'

# 初始化累积变量
total_pixels = 0
sum_normalized_pixel_values = np.zeros(3)  # 如果是RGB图像，需要三个通道的均值和方差

# 遍历文件夹中的图片文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 可根据实际情况添加其他格式
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)

            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            # print(image_path)
            # print(normalized_image_array.shape)
            # 累积归一化后的像素值和像素数量
            total_pixels += normalized_image_array.size
            sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

# 计算均值和方差
mean = sum_normalized_pixel_values / total_pixels


sum_squared_diff = np.zeros(3)
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0
            # print(normalized_image_array.shape)
            # print(mean.shape)
            # print(image_path)

            try:
                diff = (normalized_image_array - mean) ** 2
                sum_squared_diff += np.sum(diff, axis=(0, 1))
            except:
                print(f"捕获到自定义异常")
            # diff = (normalized_image_array - mean) ** 2
            # sum_squared_diff += np.sum(diff, axis=(0, 1))

variance = sum_squared_diff / total_pixels

print("Mean:", mean)
print("Variance:", variance)
