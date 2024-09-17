import torch
import torch.nn as nn
in_channel = 1
channel = 32


class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.EdgeEnhance = EdgeEnhance()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 2, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU()
        )
        self.MS1 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=(1, 1), stride=(1, 1),
                             padding=0)
        self.MS3 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=(3, 3), stride=(1, 1),
                             padding=1)
        self.MS5 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=(5, 5), stride=(1, 1),
                             padding=2)
        self.MS7 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=(7, 7), stride=(1, 1),
                             padding=3)
        self.fc11 = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=(3, 3), stride=(1, 1),
                             padding=1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Conv2d(in_channels=channel * 3, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, input):
        Fd = self.EdgeEnhance(input)
        block1 = self.fc1(input)
        block2 = torch.cat((self.MS1(block1), self.MS3(block1), self.MS7(block1), self.MS7(block1)), dim=1)
        Fm = self.fc11(block2)
        block3 = self.fc2(Fm)
        Fs = self.fc3(torch.cat((block1, block3, Fd), dim=1))
        return Fs, Fm, Fd


class EdgeEnhance(nn.Module):
    def __init__(self):
        super(EdgeEnhance, self).__init__()
        self.fc0 = nn.Conv2d(in_channels=in_channel * 2, out_channels=channel, kernel_size=(3, 3), stride=(1, 1),
                             padding=1)
        self.fc1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2d = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Sigmoid()

    def forward(self, input):
        block0 = self.fc0(input)
        block1 = self.fc1(block0)
        block2 = self.pool2d(block1)
        block3 = block2 - block0
        block4 = torch.mean(block3, dim=1, keepdim=True)
        block5 = self.fc2(block4)
        output = block5 * block0
        return output


if __name__ == "__main__":
    Encode = Encode()
    a = torch.randn(1, 2, 640, 480)
    result, result1, result2 = Encode(a)
    print(result.shape)
