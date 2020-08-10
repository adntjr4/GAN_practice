import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import matplotlib as plt

from util.summary_logging import *
from util.progress_msg import *
from model import Generator, Discriminator
from gan import GAN


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',        type=int,   default=4)
    parser.add_argument('--data_root',      type=str,   default='./data')
    parser.add_argument('--GAN',            type=str,   default='vanilla')
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--vec_len',        type=int,   default=16)
    parser.add_argument('--feature_num',    type=int,   default=16)
    parser.add_argument('--total_epoch',    type=int,   default=1024)
    parser.add_argument('--learning_rate',  type=float, default=2e-4)
    cfg = parser.parse_args()


    train_minst = torchvision.datasets.MNIST(cfg.data_root, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_minst, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_G = Generator(cfg.vec_len, cfg.feature_num).to(device)
    net_D = Discriminator(cfg.feature_num).to(device)

    fixed_vector = torch.randn(1, cfg.vec_len, 1, 1, device=device)

    optim_G = optim.Adam(net_G.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    optim_D = optim.Adam(net_D.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

    GAN = GAN(cfg.GAN_type, net_G, net_D, optim_G, optim_D, device)

    print("Starting Training Loop...")

    progmsg = ProgressMsg((cfg.total_epoch, len(train_loader)))
    lw = LossWriter()

    for epoch in range(cfg.total_epoch):
        D_error_sum = 0.0
        G_error_sum = 0.0

        for batch in enumerate(train_loader):
            GAN.epoch_train(batch)

        # after every epoch save and log print
        if epoch % 10 == 9:
            fixed_img = net_G(fixed_vector)
            tr = transforms.ToPILImage()
            fixed_img = tr(fixed_img.cpu().squeeze().detach())
            fixed_img.save('./data/tmp/%d.png'%(epoch+1))

        progmsg.line_reset()
        print('epoch [%d/%d] - D-error: %.2f, G-error: %.2f'%(epoch+1, cfg.total_epoch, D_error_sum/len(train_loader), G_error_sum/len(train_loader)))
                
        lw.write_loss('loss_d', D_error_sum/len(train_loader), epoch+1)
        lw.write_loss('loss_g', G_error_sum/len(train_loader), epoch+1)

        torch.save({'epoch': epoch+1,
                    'model_D': net_D,
                    'model_G': net_G,
                    'optimizer_D': optimizer_D,
                    'optimizer_G': optimizer_G},
                    './model/checkpoint/net_checkpoint.pth')

if __name__=='__main__':
    train()
