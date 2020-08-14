import argparse, os

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from util.summary_logging import *
from util.progress_msg import *
from model import Generator, Discriminator
from gan import GANTrainer


def train(args):
    train_minst = torchvision.datasets.MNIST(args.data_root, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_minst, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_name = './model/checkpoint/%s_checkpoint.pth'%(args.GAN_type)

    # find checkpoint and make networks
    try:
        if os.path.isfile(checkpoint_name):
            model_load = torch.load(checkpoint_name)

            net_G = model_load['model_G']
            net_D = model_load['model_D']

            optim_G = model_load['optimizer_G']
            optim_D = model_load['optimizer_D']

            start_epoch = model_load['epoch']
        else:
            net_G = Generator(args.vec_len, args.feature_num).to(device)
            net_D = Discriminator(args.feature_num).to(device)

            optim_G = optim.Adam(net_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            optim_D = optim.Adam(net_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

            start_epoch = 0
    except:
        print("error occured in loading model")

    # select GAN type
    GAN_trainer = GANTrainer(args.GAN_type, net_G, net_D, optim_G, optim_D, device, args.vec_len)

    # fixed noise for showing procedure of training
    fixed_vector = torch.randn(1, args.vec_len, 1, 1, device=device)

    print("Starting Training Loop...")

    progmsg = ProgressMsg((args.total_epoch-start_epoch, len(train_loader)))
    lw = LossWriter()

    for epoch in range(start_epoch, args.total_epoch):
        G_error_sum = 0.0
        D_error_sum = 0.0

        for batch in enumerate(train_loader):
            i, _ = batch
            G_add, D_add = GAN_trainer.epoch_train(batch)
            G_error_sum += G_add
            D_error_sum += D_add

            progmsg.print_prog_msg((epoch-start_epoch, i))

        # after every epoch save and log print
        if epoch % 10 == 9:
            fixed_img = net_G(fixed_vector)
            tr = transforms.ToPILImage()
            fixed_img = tr(fixed_img.cpu().squeeze().detach().clamp(0., 1.))
            fixed_img.save('./data/tmp/%s/%04d.png'%(args.GAN_type, epoch+1))

        progmsg.line_reset()
        print('epoch [%d/%d] - D-error: %.4f, G-error: %.4f'%(epoch+1, args.total_epoch, D_error_sum/len(train_loader), G_error_sum/len(train_loader)))
                
        lw.write_loss('loss_d', D_error_sum/len(train_loader), epoch+1)
        lw.write_loss('loss_g', G_error_sum/len(train_loader), epoch+1)

        torch.save({'epoch': epoch+1,
                    'model_D': net_D,
                    'model_G': net_G,
                    'optimizer_G': optim_G,
                    'optimizer_D': optim_D},
                    './model/checkpoint/%s_checkpoint.pth'%(args.GAN_type))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--data_root',      type=str,   default='./data')
    parser.add_argument('--GAN_type',       type=str,   default='vanilla')
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--vec_len',        type=int,   default=16)
    parser.add_argument('--feature_num',    type=int,   default=16)
    parser.add_argument('--total_epoch',    type=int,   default=2048)
    parser.add_argument('--learning_rate',  type=float, default=1e-4)
    args = parser.parse_args()

    train(args)
