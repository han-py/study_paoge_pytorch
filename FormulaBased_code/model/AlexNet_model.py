import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.RelU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f9 = nn.Linear(6*6*256, 4096)
        self.f10 = nn.Linear(4096, 4096)
        self.f11 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.RelU(self.c1(x))
        x = self.s2(x)
        x = self.RelU(self.c3(x))
        x = self.s4(x)
        x = self.RelU(self.c5(x))
        x = self.RelU(self.c6(x))
        x = self.RelU(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.RelU(self.f9(x))
        x = F.dropout(x, 0.5)
        x = self.RelU(self.f10(x))
        x = F.dropout(x, 0.5)
        x = self.f11(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (3, 227, 227)))