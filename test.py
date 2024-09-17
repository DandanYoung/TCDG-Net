import os
import torch
from PIL import Image
from model import Fusion as Net
import numpy as np
import argparse
from thop import profile
import cv2

def saveimage(x, savepath):
    x = x.cpu().numpy()[0, :, :, :]
    x = np.transpose(x, (1, 2, 0))
    x = np.resize(x, (480, 640))
    cv2.imwrite(savepath, x * 255)

def normalization(x):
    x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cpu'))
    opt = parser.parse_args()
    Net = Net().to(opt.device)
    test_data_path = './dataset/MSRS'
    Test_Image_Number = len(os.listdir(test_data_path+'/test_ir'))
    for i in range(int(Test_Image_Number)):
        channel = 1
        Test_Vis = Image.open(test_data_path + '/test_vi/' + str(i + 1) + '.png').convert('L') # visible
        Test_IR = Image.open(test_data_path + '/test_ir/' + str(i + 1) + '.png').convert('L')  # infrared image
        best_model = torch.load('./output/best_weight.pkl', map_location=lambda storage, loc: storage)
        Net.load_state_dict(best_model['weight'])
        Net.eval()
        img_test1 = np.array(Test_Vis, dtype='float32') / 255  # 将其转换为一个矩阵
        img_test2 = np.array(Test_IR, dtype='float32') / 255  # 将其转换为一个矩阵
        img_test1 = torch.from_numpy(img_test1.reshape((1, channel, img_test1.shape[0], img_test1.shape[1])))
        img_test2 = torch.from_numpy(img_test2.reshape((1, channel, img_test2.shape[0], img_test2.shape[1])))
        img_test1 = img_test1.to(opt.device)
        img_test2 = img_test2.to(opt.device)
        # flops, params = profile(Net, (img_test1, img_test2))
        # print('flops:', flops / 1000000000, 'params:', params)
        print('正在测试第%d对' % (i+1))
        with torch.no_grad():
            data_fus = Net(img_test1, img_test2)
        saveimage(data_fus, './results/' + str(i + 1) + '.png')
        print('第%d对结果已保存' % (i+1))
