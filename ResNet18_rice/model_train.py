"""
ResNet18模型训练脚本

关键函数参数说明：
=================
Data.random_split 参数说明：
- dataset: 要分割的数据集
- lengths: 包含每个分割长度的序列

Data.DataLoader 参数说明：
- dataset: 数据集
- batch_size: 每个批次的样本数量，默认为1
- shuffle: 是否打乱数据顺序，默认为False
- num_workers: 用于数据加载的子进程数，默认为0

torch.optim.Adam 参数说明：
- params: 待优化的参数
- lr: 学习率，默认为1e-3
- betas: 用于计算梯度及其平方的移动平均的系数，默认为(0.9, 0.999)
- eps: 为了数值稳定性而加到分母的小常数，默认为1e-8
- weight_decay: 权重衰减系数，默认为0

nn.CrossEntropyLoss 参数说明：
- weight: 类别权重，用于处理类别不平衡问题
- size_average: 是否对损失求均值
- ignore_index: 忽略的类别索引
- reduce: 是否降维
- reduction: 指定应用于输出的简化类型，默认为'mean'
"""

import copy
import time

import torch
# ImageFolder用于加载图像数据集，transforms用于数据预处理
from torchvision.datasets import ImageFolder
from torchvision import transforms
# Data提供数据加载工具，如DataLoader等
import torch.utils.data as Data
# numpy用于数值计算
import numpy as np
# matplotlib.pyplot用于绘制图表
import matplotlib.pyplot as plt
# 导入自定义的ResNet18模型和Residual模块
from model import ResNet18, Residual
# nn是构建神经网络的基础模块
import torch.nn as nn
# pandas用于数据处理和分析
import pandas as pd


def train_val_data_process():
    """
    处理训练和验证数据集
    
    Returns:
        tuple: (训练数据加载器, 验证数据加载器)
    """
    # 定义数据集的路径
    ROOT_TRAIN = r'data\train'

    # 定义数据标准化变换，参数为均值和标准差
    normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])
    # 定义数据集处理方法变量，包含调整大小、转换为张量和标准化
    train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
    # 加载数据集，应用指定的变换
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    # 将训练数据集按照8:2的比例随机分割为训练集和验证集
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    
    # 创建训练数据加载器
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    # 创建验证数据加载器
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    训练模型的主要过程
    
    Args:
        model: 待训练的神经网络模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        num_epochs: 训练轮数
        
    Returns:
        DataFrame: 包含训练过程中各项指标的DataFrame
    """
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    
    # 将模型放入到训练设备中
    model = model.to(device)
    
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    # 开始训练循环
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
            
        # 验证阶段
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度val_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 如果当前验证集准确率更高，则保存当前模型参数
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 选择最优参数，保存最优参数的模型
    model.load_state_dict(best_model_wts)
    # 保存最优模型参数到指定路径
    torch.save(best_model_wts, "./best_model.pth")

    # 构建训练过程记录DataFrame
    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all,})

    return train_process


def matplot_acc_loss(train_process):
    """
    绘制训练过程中的准确率和损失曲线
    
    Args:
        train_process (DataFrame): 包含训练过程记录的DataFrame
    """
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


# 主程序入口
if __name__ == '__main__':
    # 加载需要的模型
    ResNet18 = ResNet18(Residual)
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(ResNet18, train_data, val_data, num_epochs=20)
    # 绘制训练过程中的准确率和损失变化图
    matplot_acc_loss(train_process)