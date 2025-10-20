"""
FashionMNIST测试脚本
该脚本用于加载训练好的GoogLeNet模型，并在FashionMNIST测试集上进行测试
输出每个样本的预测结果和真实标签，便于观察模型表现
"""

import torch
# 导入torchvision中的transforms模块，用于图像预处理
from torchvision import transforms
# 导入FashionMNIST数据集
from torchvision.datasets import FashionMNIST
# 导入PyTorch的数据加载工具
import torch.utils.data as Data

# 导入自定义的GoogLeNet模型
from GoogLeNet.model import GoogLeNet, Inception


def test_data_process():
    """
    处理测试数据集
    下载并预处理FashionMNIST测试数据，创建数据加载器
    
    Returns:
        DataLoader: 测试数据加载器
    """
    # 加载FashionMNIST测试数据集
    # root: 数据存储路径
    # train: False表示加载测试集（True表示训练集）
    # transform: 图像预处理流程，包括调整大小为28x28和转换为Tensor
    # download: True表示自动下载数据集
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)

    # 创建数据加载器
    # batch_size: 批次大小设为1，每次处理一张图片
    # shuffle: 是否打乱数据顺序，设为True
    # num_workers: 数据加载使用的子进程数，设为0表示不使用多进程
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader

# 注释掉的代码：test_dataloader = test_data_process()


def test_model_process(model, test_dataloader):
    """
    在测试集上评估模型性能
    计算并打印模型在整个测试集上的准确率
    
    Args:
        model: 训练好的模型
        test_dataloader: 测试数据加载器
    """
    # 确定计算设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化正确预测计数和总样本数
    test_correct = 0.0
    test_num = 0

    # 关闭梯度计算以节省内存并加快计算速度
    with torch.no_grad():
        # 遍历测试集中的所有批次数据
        for test_data_x, test_data_y in test_dataloader:
            # 将数据移动到指定设备上
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式
            model.eval()

            # 前向传播得到输出结果
            output = model(test_data_x)

            # 获取预测结果（概率最大的类别）
            pre_lab = torch.argmax(output, dim=1)

            # 累计正确预测的数量
            test_correct += torch.sum(pre_lab == test_data_y.data)
            # 累计总样本数量
            test_num += test_data_x.size(0)

    # 计算并打印测试准确率
    test_acc = test_correct.double().item() / test_num
    print("测试集的准确率为：", test_acc)


# 程序入口点
if __name__ == '__main__':
    # 创建GoogLeNet模型实例
    model = GoogLeNet(Inception)
    # 加载训练好的模型权重，使用weights_only=True提高安全性
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # 获取测试数据加载器
    test_dataloader = test_data_process()
    # test_model_process(model, test_dataloader)

    # 确定计算设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义FashionMNIST数据集的类别标签
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 关闭梯度计算
    with torch.no_grad():
        # 遍历测试数据加载器中的每个样本
        for b_x, b_y in test_dataloader:
            # 将数据移到指定设备上
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为评估模式
            model.eval()

            # 前向传播得到输出
            output = model(b_x)

            # 获取预测结果（概率最大的类别索引）
            pre_lab = torch.argmax(output, dim=1)
            # 提取预测结果和真实标签的值
            result = pre_lab.item()
            label = b_y.item()

            # 打印预测结果和真实标签对应的类别名称
            print("预测结果为：", classes[result], "------", "真实标签为：", classes[label])