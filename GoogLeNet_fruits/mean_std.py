"""
计算图像数据集的均值和方差脚本

该脚本用于计算图像数据集中所有图像的像素均值和方差，
这些统计信息通常用于图像标准化处理。

numpy.zeros 参数说明：
- shape: 整数或整数序列，指定新数组的形状
- dtype: 可选，数据类型，默认为float64

numpy.sum 参数说明：
- a: 数组元素求和
- axis: 指定求和的轴向，None表示展平后求和
- dtype: 返回数组的期望数据类型

os.walk 参数说明：
- top: 根目录路径
- topdown: 可选，为True时自上而下遍历，默认为True
- onerror: 可选，错误处理函数
- followlinks: 可选，是否访问符号链接目录，默认为False
"""

from PIL import Image
import os
import numpy as np

# 文件夹路径，包含所有图片文件
folder_path = 'fruits'

# 初始化累积变量
total_pixels = 0
# 如果是RGB图像，需要三个通道的均值和方差
sum_normalized_pixel_values = np.zeros(3)

# 遍历文件夹中的图片文件，计算像素总和
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # 检查文件扩展名，只处理图像文件
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 可根据实际情况添加其他格式
            image_path = os.path.join(root, filename)
            # 使用PIL打开图像
            image = Image.open(image_path)
            # 将图像转换为numpy数组
            image_array = np.array(image)

            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            # 累积归一化后的像素值和像素数量
            total_pixels += normalized_image_array.size
            # 对图像的高和宽维度求和，保留通道维度
            sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

# 计算均值：总和除以像素总数
mean = sum_normalized_pixel_values / total_pixels

# 初始化平方差累积变量
sum_squared_diff = np.zeros(3)

# 再次遍历文件夹中的图片文件，计算方差
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)
            # 归一化像素值到0-1之间
            normalized_image_array = image_array / 255.0

            try:
                # 计算每个像素与均值的差的平方
                diff = (normalized_image_array - mean) ** 2
                # 累积平方差
                sum_squared_diff += np.sum(diff, axis=(0, 1))
            except:
                # 异常处理，打印提示信息
                print(f"捕获到自定义异常")

# 计算方差：平方差总和除以像素总数
variance = sum_squared_diff / total_pixels

# 输出计算得到的均值和方差
print("Mean:", mean)
print("Variance:", variance)