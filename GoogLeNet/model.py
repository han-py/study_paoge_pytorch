"""
GoogLeNet (Inception v1) implementation using PyTorch

卷积和池化函数参数说明：
========================
nn.Conv2d 参数说明：
- in_channels: 输入通道数
- out_channels: 输出通道数
- kernel_size: 卷积核大小
- stride: 步长，默认为1
- padding: 填充，默认为0
- dilation: 空洞卷积参数，默认为1
- groups: 分组卷积参数，默认为1
- bias: 是否使用偏置，默认为True

nn.MaxPool2d 参数说明：
- kernel_size: 池化窗口大小
- stride: 步长，默认等于kernel_size
- padding: 填充，默认为0
- dilation: 空洞池化参数，默认为1

nn.AvgPool2d 参数说明：
- kernel_size: 池化窗口大小
- stride: 步长，默认等于kernel_size
- padding: 填充，默认为0
"""

import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    """
    Inception模块实现
    该模块通过并行使用不同尺寸的卷积核来提取不同尺度的特征
    """
    def __init__(self, in_channels, c1, c2, c3, c4):
        """
        初始化Inception模块
        :param in_channels: 输入通道数
        :param c1: 路径1的1x1卷积输出通道数
        :param c2: 路径2的(1x1卷积输出通道数, 3x3卷积输出通道数)
        :param c3: 路径3的(1x1卷积输出通道数, 5x5卷积输出通道数)
        :param c4: 路径4的1x1卷积输出通道数(在3x3最大池化之后)
        """
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()

        # 路径1: 1x1卷积
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路径2: 1x1卷积 + 3x3卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        # 路径3: 1x1卷积 + 5x5卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        # 路径4: 3x3最大池化 + 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 注意最大池化和平均池化的 stride: 步长，默认等于kernel_size
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入张量
        :return: 拼接后的输出张量
        """
        # 路径1: 1x1卷积
        p1 = self.ReLU(self.p1_1(x))

        # 路径2: 1x1卷积 + 3x3卷积
        p2 = self.ReLU(self.p2_1(x))
        p2 = self.ReLU(self.p2_2(p2))

        # 路径3: 1x1卷积 + 5x5卷积
        p3 = self.ReLU(self.p3_1(x))
        p3 = self.ReLU(self.p3_2(p3))

        # 路径4: 3x3最大池化 + 1x1卷积
        p4 = self.p4_1(x)
        p4 = self.ReLU(self.p4_2(p4))

        # 在通道维度上拼接所有路径的输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) 模型实现
    """
    def __init__(self, Inception):
        """
        初始化GoogLeNet模型
        :param Inception: Inception模块类
        """
        super(GoogLeNet, self).__init__()
        # 第一个块: 7x7卷积 + 3x3最大池化
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第二个块: 1x1卷积 + 3x3卷积 + 3x3最大池化
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第三个块: 两个Inception模块 + 3x3最大池化
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第四个块: 五个Inception模块 + 3x3最大池化
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第五个块: 两个Inception模块 + 7x7平均池化 + 全连接层
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Linear(1024, 10),
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(Inception).to(device)
    print(summary(model, (1, 224, 224)))