"""
数据集划分脚本
用于将原始数据集划分为训练集和测试集
"""

import os
# shutil.copy函数用于文件复制操作
from shutil import copy
# random模块用于随机抽样
import random


def mkfile(file):
    """
    创建文件夹的辅助函数
    
    Args:
        file (str): 要创建的文件夹路径
    """
    if not os.path.exists(file):
        os.makedirs(file)


# 获取原始数据文件夹下所有类别文件夹名（即需要分类的类名）
# 注意：虽然变量名为flower_class，但实际处理的是水果图片分类
file_path = 'Rice_Image_Dataset'
flower_class = [cla for cla in os.listdir(file_path)]

# 创建训练集train文件夹，并为每个类别在其目录下创建子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

# 创建测试集test文件夹，并为每个类别在其目录下创建子目录
mkfile('data/test')
for cla in flower_class:
    mkfile('data/test/' + cla)

# 划分比例，训练集 : 测试集 = 9 : 1
split_rate = 0.1

# 遍历所有类别的全部图像并按比例分成训练集和测试集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
    images = os.listdir(cla_path)  # images列表存储了该目录下所有图像的名称
    num = len(images)
    # 从images列表中随机抽取k个图像名称作为测试集
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        # eval_index中保存测试集的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/test/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到测试集路径
        
        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        # 显示处理进度条
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()

print("processing done!")