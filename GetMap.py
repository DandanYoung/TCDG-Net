import torch
import torch.nn as nn

device = torch.device('cpu')
# device = torch.device('cuda:1')
def sobel_conv(data, channel=1):
    conv_op_x = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [ 0, 0, 0],
                                   [ 1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5 * abs(edge_x) + 0.5 * abs(edge_y)
    return result


def Laplace_conv(data, channel=1):
    conv_op_x = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    Laplace_kernel_x = torch.tensor([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    Laplace_kernel_y = torch.tensor([[-1, -1, -1],
                                   [ -1, 8, -1],
                                   [ -1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).to(device)
    conv_op_x.weight.data = Laplace_kernel_x
    conv_op_y.weight.data = Laplace_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5 * abs(edge_x) + 0.5 * abs(edge_y)
    return result


def prewitt_conv(data, channel=1):
    conv_op_x = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, (3, 3), stride=(1, 1), padding=1, groups=channel, bias=False)
    Prewitt_kernel_x = torch.tensor([[1, 1, 1],
                                   [ 0, 0, 0],
                                   [ -1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1,
                                                                                                         1).to(device)
    Prewitt_kernel_y = torch.tensor([[-1, 0, 1],
                                     [-1, 0, 1],
                                     [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1,
                                                                                                        1).to(device)
    conv_op_x.weight.data = Prewitt_kernel_x
    conv_op_y.weight.data = Prewitt_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5 * abs(edge_x) + 0.5 * abs(edge_y)
    return result


def GetGradMap(img1, img2):
    m1 = sobel_conv(img1)
    m1 = m1/torch.max(m1)
    m2 = sobel_conv(img2)
    m2 = m2/torch.max(m2)
    f = (1 - m1) * (1 - m2) + m1 * m2
    return 1-f

if __name__ == "__main__":
    a = torch.randn(2, 1, 140, 130)
    b = torch.randn(2, 1, 140, 130)
    result = GetGradMap(a, b)
    print(result.shape)


