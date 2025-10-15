import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data

from LeNet.model import LeNet


def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader

# test_dataloader = test_data_process()


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_correct = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)

            test_correct += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    test_acc = test_correct.double().item() / test_num
    print("测试集的准确率为：", test_acc)


if __name__ == '__main__':
    model = LeNet()
    # 添加weights_only=True参数以避免FutureWarning
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)