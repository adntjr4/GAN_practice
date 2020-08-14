import torch
import torch.nn as nn

class GANTrainer():
    def __init__(self, GAN_type, net_G, net_D, optim_G, optim_D, device, vec_len):
        self.GAN_type = GAN_type
        self.net_G = net_G
        self.net_D = net_D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.device = device
        self.vec_len = vec_len

        assert self.GAN_type in ['vanilla', 'WGAN', 'WGAN_GP' ,'SNGAN', 'LSGAN']

        if self.GAN_type == 'vanilla':
            self.criterion = nn.BCELoss()
        elif self.GAN_type == 'WGAN':
            self.criterion = None
        elif self.GAN_type == 'WGAN_GP':
            self.criterion = None
        elif self.GAN_type == 'SNGAN':
            self.criterion = None
        elif self.GAN_type == 'LSGAN':
            self.criterion = None

    def epoch_train(self, batch):
        if self.GAN_type == 'vanilla':
            return self.vanilla_train(batch)
        elif self.GAN_type == 'WGAN':
            return self.WGAN_train(batch)
        elif self.GAN_type == 'WGAN_GP':
            return self.WGAN_GP_train(batch)
        elif self.GAN_type == 'SNGAN':
            return self.SNGAN_train(batch)
        elif self.GAN_type == 'LSGAN':
            return self.LSGAN_train(batch)

    # vanilla GAN
    def vanilla_train(self, batch):
        _real = 1.
        _fake = 0.

        i, data = batch
        
        # update D network (train real & fake batch)
        self.net_D.zero_grad()

        real_img = data[0].to(self.device)
        b_size = real_img.size(0)

        real_label = torch.full((b_size,), _real, device=self.device)
        real_discrimination = self.net_D(real_img).view(-1)
        real_D_error = self.criterion(real_discrimination, real_label)
        real_D_error.backward()

        rand_seed_vector = torch.randn(b_size, self.vec_len, 1, 1, device=self.device)
        fake_img = self.net_G(rand_seed_vector)
        fake_label = torch.full((b_size,), _fake, device=self.device)
        fake_discrimination = self.net_D(fake_img.detach()).view(-1)
        fake_D_error = self.criterion(fake_discrimination, fake_label)
        fake_D_error.backward()

        self.optim_D.step()

        # update G network (train fake batch)
        self.net_G.zero_grad()

        fake_all_output = self.net_D(fake_img).view(-1)
        fake_G_error = self.criterion(fake_all_output, real_label)
        fake_G_error.backward()
        self.optim_G.step()

        # output training stats
        D_error = real_D_error.item() + fake_D_error.item()
        G_error = fake_G_error.item() 
        return D_error, G_error

    # WGAN
    def WGAN_train(self, batch, G_net, D_net, G_op, D_op, device):
        pass

    # WGAN-GP
    def WGAN_gp_train(self, batch, G_net, D_net, G_op, D_op, device):
        pass

    # SNGAN
    def SNGAN_train(self, batch, G_net, D_net, G_op, D_op, device):
        pass

    # LSGAN
    def LSGAN_train(self, batch, G_net, D_net, G_op, D_op, device):
        pass