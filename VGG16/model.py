import torch
from torch import nn
from torchsummary import summary


class VGG16(nn.Module):
    """
    VGG16网络模型实现
    这是一个简化版的VGG16网络，用于处理单通道图像输入（如MNIST数据集）
    """

    def __init__(self):
        """初始化VGG16网络结构"""
        super(VGG16, self).__init__()
        
        # 第一个卷积块：包含两个卷积层和一个最大池化层
        # 输入通道: 1, 输出通道: 64
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块：包含两个卷积层和一个最大池化层
        # 输入通道: 64, 输出通道: 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块：包含三个卷积层和一个最大池化层
        # 输入通道: 128, 输出通道: 256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第四个卷积块：包含三个卷积层和一个最大池化层
        # 输入通道: 256, 输出通道: 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第五个卷积块：包含三个卷积层和一个最大池化层
        # 输入通道: 512, 输出通道: 512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层块：包含展平层、全连接层和Dropout层
        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),  # 经过5次池化后，224x224变为7x7
            nn.ReLU(),
            nn.Dropout(0.5),              # Dropout防止过拟合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),              # Dropout防止过拟合
            nn.Linear(128, 10)   # 因为类别较小，神经元个数可以不需要太多
        )

        # 参数初始化
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # 对卷积层使用kaiming初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对全连接层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入张量，形状为(batch_size, 1, 224, 224)
            
        Returns:
            输出张量，形状为(batch_size, 10)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

if __name__ == "__main__":
    # 检查是否有可用的GPU，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型实例并移至指定设备
    model = VGG16().to(device)
    # 打印模型结构摘要信息
    print(summary(model, (1, 224, 224)))  # 输入图像大小为224x224，单通道