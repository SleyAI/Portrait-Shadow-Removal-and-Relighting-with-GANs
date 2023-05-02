import torch
import argparse
from models.psp import pSp
from models.psp2 import pSp2
from models.encoders.psp_encoders import Encoder4Editing
from models.encoders.psp_e4e_encoders import Encoder4Editing2

import sys
sys.path.extend(['.', '..'])

from models.stylegan2.model import Generator


def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    is_cars = 'car' in opts['dataset_type']
    is_faces = 'ffhq' in opts['dataset_type']
    if is_faces:
        opts['stylegan_size'] = 1024
    elif is_cars:
        opts['stylegan_size'] = 1024
    else:
        opts['stylegan_size'] = 1024

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts['is_train'] = False
    opts = argparse.Namespace(**opts)

    # net = pSp(opts)
    net = pSp2(opts)
    net.eval()
    net = net.to(device)
    return net, opts


def setup_e4e_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp2(opts)
    net.eval()
    net = net.to(device)
    return net, opts


def load_generator(checkpoint_path, device='cuda'):
    generator = Generator(1024, 512, 8, channel_multiplier=2)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(ckpt['g_ema'])
    generator.eval()
    generator.to(device)
    return generator


def load_e4e_standalone(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = argparse.Namespace(**ckpt['opts'])
    e4e = Encoder4Editing2(50, 'ir_se', opts)
    e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    e4e.load_state_dict(e4e_dict)
    e4e.eval()
    e4e = e4e.to(device)
    latent_avg = ckpt['latent_avg'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

    e4e.register_forward_hook(add_latent_avg)
    return e4e
