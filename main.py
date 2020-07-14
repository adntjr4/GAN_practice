import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from util.progress_msg import *
from model import Generator, Discriminator

num_workers = 4
mnist_root = './data'
batch_size = 128
vec_len = 16
feature_num =  16
total_epoch = 100
learning_rate = 0.0002
beta1 = 0.5

train_minst = torchvision.datasets.MNIST(mnist_root, train=True, transform=transforms.ToTensor(), download=True)
test_minst = torchvision.datasets.MNIST(mnist_root, train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_minst, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_minst, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

net_G = Generator(vec_len, feature_num).to(device)
net_D = Discriminator(feature_num).to(device)
net_G.apply(weights_init)
net_D.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, vec_len, 1, 1, device=device)
_real = 1.
_fake = 0.

optimizer_G = optim.Adam(net_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(net_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))

print("Starting Training Loop...")

progmsg = ProgressMsg((total_epoch, math.ceil(60000/batch_size)))

for epoch in range(total_epoch):
    for i, data in enumerate(train_loader):
        # update D network (train real & fake batch)
        net_D.zero_grad()

        real_img = data[0].to(device)
        b_size = real_img.size(0)

        real_label = torch.full((b_size,), _real, device=device)
        real_discrimination = net_D(real_img).view(-1)
        real_D_error = criterion(real_discrimination, real_label)
        real_D_error.backward()

        rand_seed_vector = torch.randn(b_size, vec_len, 1, 1, device=device)
        fake_img = net_G(rand_seed_vector)
        fake_label = torch.full((b_size,), _fake, device=device)
        fake_discrimination = net_D(fake_img.detach()).view(-1)
        fake_D_error = criterion(fake_discrimination, fake_label)
        fake_D_error.backward()

        optimizer_D.step()

        # update G network (train fake batch)
        net_G.zero_grad()

        fake_whole_output = net_D(fake_img).view(-1)
        fake_G_error = criterion(fake_whole_output, real_label)
        fake_G_error.backward()
        optimizer_G.step()

        # output training stats
        progmsg.print_prog_msg((epoch, i))
        #if i % 50 == 0:
            
