import argparse
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from model import Fusion as Net1
import time
from loss import Fusionloss

if __name__ == '__main__':
    batch_size = 32
    epochs = 200
    lr = 1e-4
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cpu'))
    opt = parser.parse_args()
    train_start_time = time.time()
    train_data_path = './dataset/MSRS/'
    val_data_path = './dataset/MSRS/'
    train_root_1 = train_data_path + 'train_vi'
    train_root_2 = train_data_path + 'train_ir'
    val_root_1 = val_data_path + 'val_vi'
    val_root_2 = val_data_path + 'val_ir'
    train_path = './output/'
    resume = train_path + 'best_weight.pkl'
    Train_Image_Number = len(os.listdir(train_root_1 + '/vi_crop'))
    Val_Image_Number = len(os.listdir(val_root_1 + '/vi_crop'))
    transforms = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
    Net = Net1().to(opt.device)

    # =============================================================================
    # Preprocessing and dataset establishment
    # ============================================================================
    train_Data_1 = torchvision.datasets.ImageFolder(train_root_1, transform=transforms)
    train_dataloader_1 = torch.utils.data.DataLoader(train_Data_1, batch_size, shuffle=False)
    train_Data_2 = torchvision.datasets.ImageFolder(train_root_2, transform=transforms)
    train_dataloader_2 = torch.utils.data.DataLoader(train_Data_2, batch_size, shuffle=False)

    val_Data_1 = torchvision.datasets.ImageFolder(val_root_1, transform=transforms)
    val_dataloader_1 = torch.utils.data.DataLoader(val_Data_1, batch_size, shuffle=False)
    val_Data_2 = torchvision.datasets.ImageFolder(val_root_2, transform=transforms)
    val_dataloader_2 = torch.utils.data.DataLoader(val_Data_2, batch_size, shuffle=False)

    train_Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
    val_Iter_per_epoch = (Val_Image_Number % batch_size != 0) + Val_Image_Number // batch_size

    # =============================================================================
    optimizer1 = optim.Adam(Net.parameters(), lr=lr)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [1000], gamma=0.1)
    L1Loss = nn.L1Loss()
    Lg_loss = Fusionloss()
    best_loss = 10
    start_epoch = 0
    best_epoch = start_epoch

    best_weights = copy.deepcopy(Net.state_dict())
    # =============================================================================
    print('============ Training Begins ===============')
    for iteration in range(start_epoch, epochs):
        train_loss = 0
        l1 = 0
        lg = 0
        print("epochs:", iteration + 1)
        train_data_iter_1 = iter(train_dataloader_1)
        train_data_iter_2 = iter(train_dataloader_2)
        val_data_iter_1 = iter(val_dataloader_1)
        val_data_iter_2 = iter(val_dataloader_2)

        for step in range(train_Iter_per_epoch):
            data_1, _ = next(train_data_iter_1)
            data_2, _ = next(train_data_iter_2)
            data_1 = data_1.to(opt.device)
            data_2 = data_2.to(opt.device)
            optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # =====================================================================
            # Calculate loss
            # =====================================================================
            data_fus = Net(data_1, data_2)
            loss1 = 0.5 * L1Loss(data_1, data_fus) + 0.5 * L1Loss(data_2, data_fus)
            Lg = Lg_loss(data_1, data_2, data_fus)
            loss = 0.2 * loss1 + 0.8 * Lg
            loss.backward()
            optimizer1.step()
            los = loss.item()
            train_loss += los
            l1 += loss1.item()
            lg += Lg.item()
            if step % 100 == 0:
                print("step:", step, "/", Train_Image_Number//batch_size)

        l1_epoch_loss = l1/train_Iter_per_epoch
        lg_epoch_loss = lg/train_Iter_per_epoch
        train_epoch_loss = train_loss/train_Iter_per_epoch
        print('train total loss: {:.6f}'.format(train_epoch_loss))
        print('train l1 loss: {:.6f}'.format(l1_epoch_loss))
        print('train lg loss: {:.6f}'.format(lg_epoch_loss))

        with torch.no_grad():
            val_loss = 0
            for i in range(val_Iter_per_epoch):
                data_1, _ = next(val_data_iter_1)
                data_2, _ = next(val_data_iter_2)
                data_1 = data_1.to(opt.device)
                data_2 = data_2.to(opt.device)
                optimizer1.zero_grad()
                # =====================================================================
                # Calculate loss
                # =====================================================================
                data_fus = Net(data_1, data_2)
                loss1 = 0.5 * L1Loss(data_1, data_fus) + 0.5 * L1Loss(data_2, data_fus)
                Lg = Lg_loss(data_1, data_2, data_fus)
                loss = 0.2 * loss1 + 0.8 * Lg
                los = loss.item()
                val_loss += los
            val_epoch_loss = val_loss/val_Iter_per_epoch
            print('val total loss: {:.6f}'.format(val_epoch_loss))

        if val_epoch_loss < best_loss:
            best_epoch = iteration + 1
            best_loss = val_epoch_loss
            best_weights = copy.deepcopy(Net.state_dict())
            torch.save({'weight': best_weights,
                        'epoch': iteration + 1,
                        'loss': train_epoch_loss,
                        'optimizer_state_dict': optimizer1.state_dict()},
                       os.path.join(train_path, 'best_weight.pkl'))

        best_epoch = iteration + 1
        best_weights = copy.deepcopy(Net.state_dict())

        if (iteration + 1) % 50 == 0:
            weights = copy.deepcopy(Net.state_dict())
            torch.save({'weight': best_weights,
                        'epoch': iteration + 1,
                        'loss': train_epoch_loss,
                        'optimizer_state_dict': optimizer1.state_dict()},
                       os.path.join(train_path, f'weight_{iteration+1}.pkl'))
        print('best epoch:{:.0f}'.format(best_epoch))
        scheduler1.step()
        train_end_time = (time.time()-train_start_time) / 3600
        print(f'train time : {train_end_time :.4f} hour')

