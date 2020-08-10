import torch
import torch.nn as nn

class GAN():
    def __init__(self, GAN_type, net_G, net_D, optim_G, optim_D, device):
        self.GAN_type = GAN_type
        self.net_G = net_G
        self.net_D = net_D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.device = device

        if self.GAN_type = 'vanilla':
            self.criterion = nn.BCELoss()

    def epoch_train(self, batch):
        if self.GAN_type = 'vanilla':
            self.vanilla_GAN_train(batch)

    # vanilla GAN
    def vanilla_GAN_train(self, batch):
        _real = 1.
        _fake = 0.

        i, data = batch
        
        # update D network (train real & fake batch)
        self.net_D.zero_grad()

        real_img = data[0].to(device)
        b_size = real_img.size(0)

        real_label = torch.full((b_size,), _real, device=device)
        real_discrimination = self.net_D(real_img).view(-1)
        real_D_error = self.criterion(real_discrimination, real_label)
        real_D_error.backward()

        rand_seed_vector = torch.randn(b_size, cfg.vec_len, 1, 1, device=device)
        fake_img = self.net_G(rand_seed_vector)
        fake_label = torch.full((b_size,), _fake, device=device)
        fake_discrimination = self.net_D(fake_img.detach()).view(-1)
        fake_D_error = self.criterion(fake_discrimination, fake_label)
        fake_D_error.backward()

        self.optim_D.step()

        # update G network (train fake batch)
        self.net_G.zero_grad()

        fake_whole_output = self.net_D(fake_img).view(-1)
        fake_G_error = self.criterion(fake_whole_output, real_label)
        fake_G_error.backward()
        self.optim_G.step()

        # output training stats
        D_error = real_D_error.item() + fake_D_error.item()
        G_error = fake_G_error.item() 
        return D_error, G_error

    # WGAN
    def WGAN_train(batch, G_net, D_net, G_op, D_op, device):
        pass

    # WGAN-GP
    def WGAN_gp_train(batch, G_net, D_net, G_op, D_op, device):
        pass

    # SNGAN
    def SNGAN_train(batch, G_net, D_net, G_op, D_op, device):
        pass

    # LSGAN
    def LSGAN_train(batch, G_net, D_net, G_op, D_op, device):
        pass