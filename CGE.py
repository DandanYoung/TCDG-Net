import torch
import torch.nn as nn

channel = 32
class LIP(nn.Module):
    def __init__(self):
        super(LIP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.fc1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.fc2_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.fc1_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.fc2_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.fc4 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.fc5 =  nn.Sequential(
            nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, fea1, fea2):
        block_l_1 = self.fc1(fea1) * fea2 + fea1
        block_r_1 = self.fc2(fea2) * fea1 + fea2
        block_l_1_1 = self.fc1_1(block_l_1) * block_r_1 + block_l_1
        block_r_1_1 = self.fc2_1(block_r_1) * block_l_1 + block_r_1
        block_l_1_2 = self.fc1_2(block_l_1_1) * block_r_1_1 + block_l_1_1
        block_r_1_2 = self.fc2_2(block_r_1_1) * block_l_1_1 + block_r_1_1

        block_l_2 = self.fc3(block_l_1_2) + block_l_1_2
        block_r_2 = self.fc4(block_r_1_2) + block_r_1_2
        block = self.fc5(torch.cat((block_l_2, block_r_2), dim=1))
        return block


if __name__ == "__main__":
    LIP = LIP()
    a = torch.randn(2, 32, 120, 130)
    b = torch.randn(2, 32, 120, 130)
    fea = LIP(a, b)
    print(fea.shape)
