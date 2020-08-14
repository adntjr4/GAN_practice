
import argparse, os

import torch
from torchvision import transforms


def test(args, image_num=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_name = './model/checkpoint/%s_checkpoint.pth'%(args.GAN_type)

    # find checkpoint and make networks
    try:
        if os.path.isfile(checkpoint_name):
            model_load = torch.load(checkpoint_name)

            net_G = model_load['model_G']
    except:
        print("error occured in loading model")

    print("Generating images...")

    for i in range(image_num):
        noise = torch.randn(1, args.vec_len, 1, 1, device=device)
        generated_tensor = net_G(noise)
        tr = transforms.ToPILImage()
        generated_img = tr(generated_tensor.cpu().squeeze().detach().clamp(0., 1.))
        generated_img.save('./data/result/%s/%03d.png'%(args.GAN_type, i+1))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--GAN_type',       type=str,   default='vanilla')
    parser.add_argument('--vec_len',        type=int,   default=16)
    args = parser.parse_args()

    test(args)
