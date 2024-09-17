import torch
import torch.nn as nn
from GetMap import GetGradMap
from SFE import Encode
from CGE import LIP
from TGE import GIP
import torch.nn.functional as F
channel = 32
from thop import profile
from thop import clever_format

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.GetMap = GetGradMap
        self.encode = Encode()
        self.LIP = LIP()
        self.GIP = GIP()
        self.fc1 = nn.Conv2d(2, 1, 3, 1, 1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, vis, infra):
        mask = self.GetMap(vis, infra)
        input = torch.cat((vis, infra), dim=1)
        Fs, Fm, Fd = self.encode(input)
        F_m = Fm * (1 - mask)
        F_d = Fd * mask
        Ec = self.LIP(Fs, F_m)
        size = [Fs.shape[2], Fs.shape[3]]
        fea1 = F.interpolate(Fs, scale_factor=0.25, mode='nearest')
        fea2 = F.interpolate(F_d, scale_factor=0.25, mode='nearest')
        map_size = [fea1.shape[2], fea1.shape[3]]
        mask1 = F.interpolate(mask, size=map_size, mode='nearest')
        fea_g = self.GIP(fea1, fea2, mask1, size) + Fs + F_d
        Et = self.fc2(fea_g)
        fea_out = (1-mask) * Ec + mask * Et
        return fea_out
