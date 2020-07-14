import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    10 length vector in, 1x28x28 image out
    '''
    def __init__(self, vec_len, feature_num):
        super().__init__()
        self.layer1 = GeneratorBlock(vec_len, feature_num*4, 4, 1, 0)
        self.layer2 = GeneratorBlock(feature_num*4, feature_num*2, 4, 2, 1)
        self.layer3 = GeneratorBlock(feature_num*2, feature_num*1, 4, 2, 1)
        self.last_conv = nn.ConvTranspose2d(feature_num, 1, 4, 2, 3, bias=False)
        self.act_tanh = nn.Tanh()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last_conv(out)
        out = self.act_tanh(out)
        return out

class GeneratorBlock(nn.Module):
    '''
    basic block of generator which include transpose covolutional layer, BN, activation func
    '''
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv_layer = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.BN = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(True)
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.BN(out)
        out = self.act(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.layer1 = DiscriminatorBlock(1, feature_num, 4, 2, 3)
        self.layer2 = DiscriminatorBlock(feature_num*1, feature_num*2, 4, 2, 1)
        self.layer3 = DiscriminatorBlock(feature_num*2, feature_num*4, 4, 2, 1)
        self.last_conv = nn.Conv2d(feature_num*4, 1, 4, 1, 0, bias=False)
        self.act_sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last_conv(out)
        out = self.act_sigmoid(out)
        return out
        
class DiscriminatorBlock(nn.Module):
    '''
    basic block of discriminator which include convolutional layer, BN, activation func
    '''
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.BN = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, True)
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.BN(out)
        out = self.act(out)
        return out

if __name__ == '__main__':
    t_c = Discriminator(10)
    tt = torch.randn((10, 1, 28, 28))
    oo = t_c(tt)
    print(oo.shape)

