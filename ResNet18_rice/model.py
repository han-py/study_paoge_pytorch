"""
ResNet16网络模型实现
这是一个简化版的ResNet18网络，用于处理单通道图像输入（如MNIST数据集、FashionMNIST数据集）

关键函数参数说明：
=================
nn.Conv2d 参数说明：
- in_channels: 输入通道数
- out_channels: 输出通道数
- kernel_size: 卷积核大小
- stride: 步长，默认为1
- padding: 填充，默认为0

nn.BatchNorm2d 参数说明：
- num_features: 特征维度，与输出通道数相同

nn.MaxPool2d 参数说明：
- kernel_size: 池化核大小
- stride: 步长
- padding: 填充

nn.AdaptiveAvgPool2d 参数说明：
- output_size: 输出尺寸，可以是单个数字或元组
"""

import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    """
    残差块实现
    通过跳跃连接解决深度网络中的梯度消失问题
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        """
        初始化残差块
        :param input_channels: 输入通道数
        :param num_channels: 输出通道数
        :param use_1x1conv: 是否使用1x1卷积调整维度
        :param strides: 步长
        """
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        # 第一个3x3卷积层
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        # 第二个3x3卷积层
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        # 第一个批归一化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        # 第二个批归一化层
        self.bn2 = nn.BatchNorm2d(num_channels)
        # 1x1卷积层用于调整维度（当输入输出通道数不一致时使用）
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入张量
        :return: 输出张量
        """
        # 第一条路径：卷积 -> 批归一化 -> ReLU -> 卷积 -> 批归一化
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        # 第二条路径：恒等映射或1x1卷积（用于维度匹配）
        if self.conv3:
            x = self.conv3(x)

        # 跳跃连接：将两条路径的结果相加后通过ReLU激活函数
        return self.ReLU(y + x)


class ResNet18(nn.Module):
    """
    ResNet18网络模型实现
    由多个残差块组成，有效解决了深度网络训练困难的问题
    """
    def __init__(self, Residual):
        """
        初始化ResNet18网络结构
        :param Residual: 残差块类
        """
        super(ResNet18, self).__init__()
        # 第一个块：7x7卷积 + 批归一化 + ReLU + 3x3最大池化
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第二个块：两个残差块（输出通道数64）
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1x1conv=False, strides=1),
            Residual(64, 64, use_1x1conv=False, strides=1)
        )
        # 第三个块：两个残差块（输出通道数从64增加到128）
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128, use_1x1conv=False, strides=1)
        )
        # 第四个块：两个残差块（输出通道数从128增加到256）
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256, use_1x1conv=False, strides=1)
        )
        # 第五个块：两个残差块（输出通道数从256增加到512）
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, strides=2),
            Residual(512, 512, use_1x1conv=False, strides=1)
        )
        # 第六个块：自适应平均池化 + Flatten + 全连接层
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # (1, 1)是元组
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    print(summary(model, (1, 224, 224)))