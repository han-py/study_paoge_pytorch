"""
ResNet18模型测试脚本

该脚本用于测试训练好的ResNet18模型在猫狗分类任务上的性能。

关键函数参数说明：
=================
transforms.Normalize 参数说明：
- mean: 每个通道的均值
- std: 每个通道的标准差
- inplace: 是否原地操作，默认为False

transforms.Resize 参数说明：
- size: 输出尺寸，可以是整数或元组
- interpolation: 插值方法，默认为双线性插值

transforms.ToTensor 参数说明：
- 将PIL Image或numpy.ndarray转换为tensor并归一化到[0,1]

torch.load 参数说明：
- f: 保存文件的路径
- map_location: 指定加载设备位置
- pickle_module: 用于解封的pickle模块，默认为pickle

torch.argmax 参数说明：
- input: 输入张量
- dim: 要计算最大值的维度
- keepdim: 输出是否保留维度
"""

import torch
# Data提供数据加载工具，如DataLoader等
import torch.utils.data as Data
# transforms用于图像预处理
from torchvision import transforms
# FashionMNIST为示例数据集（虽然本项目未使用）
from torchvision.datasets import FashionMNIST
# 导入自定义的ResNet18模型和Residual模块
from model import ResNet18, Residual
# ImageFolder用于加载图像数据集
from torchvision.datasets import ImageFolder
# PIL用于图像处理
from PIL import Image


def test_data_process():
    """
    处理测试数据集
    
    Returns:
        DataLoader: 测试数据加载器
    """
    # 定义数据集的路径
    ROOT_TRAIN = r'data\test'

    # 定义数据标准化变换，参数为均值和标准差
    normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])
    # 定义数据集处理方法变量，包含调整大小、转换为张量和标准化
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集，应用指定的变换
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)

    # 创建测试数据加载器
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    """
    测试模型性能
    
    Args:
        model: 训练好的模型
        test_dataloader: 测试数据加载器
    """
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 将模型放入到测试设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)


# 主程序入口
if __name__ == "__main__":
    # 加载模型
    model = ResNet18(Residual)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    # 利用现有的模型进行模型的测试
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # 定义类别标签
    classes = ['mask', 'no_mask']
    
    # # 逐个样本进行测试并输出预测结果和真实标签
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         # 设置模型为验证模型
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测值：",  classes[result], "------", "真实值：", classes[label])

    # 加载单张图片进行测试
    image = Image.open('no_mask.jfif')

    # 定义图像预处理方法
    normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])
    # 定义数据集处理方法变量，包含调整大小、转换为张量和标准化
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)

    # 添加批次维度，因为模型需要批次输入
    image = image.unsqueeze(0)

    # 进行预测
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测值：",  classes[result])