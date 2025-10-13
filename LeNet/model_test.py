import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data


def test_data_process():
    test_data = FashionMNIST(root='./data',
                              test=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader