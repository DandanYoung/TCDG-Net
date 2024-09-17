import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import transformer
channel = 16
patch_size = 8
stride = 8


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    # paddings = (0, 0, 0, 0)
    # images, paddings = same_padding(images, ksizes, strides, rates)
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=1, stride=strides, padding=0)
    patches = unfold(images)
    return patches


class FS(nn.Module):
    def __init__(self):
        super(FS, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, fea1, fea2, mask):
        T = torch.mean(mask)
        patch_fea1 = extract_image_patches(fea1, ksizes=[self.patch_size, self.patch_size],
                                                       strides=[self.stride, self.stride], rates=[1, 1], padding=0)
        patch_fea1 = patch_fea1.permute(0, 2, 1)
        patch_fea2 = extract_image_patches(fea2, ksizes=[self.patch_size, self.patch_size],
                                           strides=[self.stride, self.stride], rates=[1, 1], padding=0)
        patch_fea2 = patch_fea2.permute(0, 2, 1)

        patch_mask = extract_image_patches(mask, ksizes=[self.patch_size, self.patch_size],
                                                         strides=[self.stride, self.stride], rates=[1, 1],
                                                         padding=0)
        patch_mask = patch_mask.permute(0, 2, 1)
        patch_mask = torch.mean(patch_mask, dim=2).unsqueeze(dim=-2)
        patch_mask[patch_mask <= T] = 0.0
        return patch_fea1, patch_fea2, patch_mask


class GIP(nn.Module):
    def __init__(self):
        super(GIP, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.size = channel * patch_size * patch_size
        self.FS = FS()
        self.fc1 = nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.fc2 = nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.transformer1 = transformer(d_model=self.size, d_inner=2 * self.size, n_layers=1)
        self.transformer2 = transformer(d_model=self.size, d_inner=2 * self.size, n_layers=1)
        self.fc3 = nn.Sequential(
            nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

    def forward(self, in_fea1, in_fea2, map, size):
        in_fea1 = self.fc1(in_fea1)
        in_fea2 = self.fc2(in_fea2)
        feat1_1, feat2_1, mask = self.FS(in_fea1, in_fea2, map)
        gf1 = self.transformer1(feat1_1, src_mask=mask)
        gf2 = self.transformer2(feat2_1, src_mask=mask)
        gf1 = gf1.permute(0, 2, 1)
        gf2 = gf2.permute(0, 2, 1)
        fea1_unfold = torch.nn.functional.fold(gf1, (in_fea1.shape[2], in_fea1.shape[3]),
                                                (self.patch_size, self.patch_size),
                                                padding=0, stride=self.stride)
        fea2_unfold = torch.nn.functional.fold(gf2, (in_fea2.shape[2], in_fea2.shape[3]),
                                                (self.patch_size, self.patch_size),
                                                padding=0, stride=self.stride)
        feat_out = torch.cat((fea1_unfold, fea2_unfold), dim=1)
        feat_out = F.interpolate(feat_out, size=size, mode='nearest')
        output = self.fc3(feat_out)
        return output