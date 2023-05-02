from datasets.shadow_detection_data_loader import make_datapath_list, ImageDataset, ImageTransform, ImageTransformOwn
from models.st_cgan.ST_CGAN import Generator
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

import argparse
import torch
import os
import sys

torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_parser():
    parser = argparse.ArgumentParser(
        prog='ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal',
        usage='python3 main.py',
        description='This module demonstrates shadow detection and removal using ST-CGAN.',
        add_help=True)

    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-i', '--image_path', type=str, default=None, help='file path of image you want to test')
    parser.add_argument('-o', '--out_path', type=str, default='./test_result', help='saving path')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-rs', '--resized_size', type=int, default=256)

    return parser


def fix_model_state_dict(state_dict):
    '''
    remove 'module.' of dataparallel
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def check_dir():
    if not os.path.exists('./test_result'):
        os.mkdir('./test_result')
    if not os.path.exists('./test_result/detected_shadow'):
        os.mkdir('./test_result/detected_shadow')
    if not os.path.exists('./test_result/shadow_removal_image'):
        os.mkdir('./test_result/shadow_removal_image')
    if not os.path.exists('./test_result/grid'):
        os.mkdir('./test_result/grid')


def unnormalize(x):
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


def test(G1, G2, test_dataset):
    '''
    this module test dataset from ISTD dataset
    '''
    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    G2.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        G2 = torch.nn.DataParallel(G2)
        print("parallel mode")

    print("device:{}".format(device))

    G1.eval()
    G2.eval()

    for n, (img, gt_shadow, gt) in enumerate([test_dataset[i] for i in range(test_dataset.__len__())]):
        print(test_dataset.img_list['path_A'][n].split('/')[4][:-4])

        img = torch.unsqueeze(img, dim=0)
        gt_shadow = torch.unsqueeze(gt_shadow, dim=0)
        gt = torch.unsqueeze(gt, dim=0)

        with torch.no_grad():
            detected_shadow = G1(img.to(device))
            detected_shadow = detected_shadow.to(torch.device('cpu'))
            concat = torch.cat([img, detected_shadow], dim=1)
            shadow_removal_image = G2(concat.to(device))
            shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

        grid = make_grid(torch.cat([unnormalize(img), unnormalize(gt), unnormalize(shadow_removal_image),
                                    unnormalize(torch.cat([gt_shadow, gt_shadow, gt_shadow], dim=1)),
                                    unnormalize(torch.cat([detected_shadow, detected_shadow, detected_shadow], dim=1))],
                                   dim=0))

        save_image(grid, './test_result/grid/' + test_dataset.img_list['path_A'][n].split('/')[4])

        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow.save('./test_result/detected_shadow/' + test_dataset.img_list['path_A'][n].split('/')[4])

        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
        shadow_removal_image.save(
            './test_result/shadow_removal_image/' + test_dataset.img_list['path_A'][n].split('/')[4])


def detect_shadows(path, out_path, size, img_transform):
    dirname = os.getcwd()
    print('Looking for shadows in generated images and selecting best one ...')
    G1 = Generator(input_channels=3, output_channels=1)

    G1_weights = torch.load(os.path.join(dirname, 'models/st_cgan/ST-CGAN_G1_' + '1500.pth'))
    G1.load_state_dict(fix_model_state_dict(G1_weights))

    latents_dir = path + '_latent_images.npy'
    latents = np.load(latents_dir, allow_pickle=True)

    # for image in tqdm(sorted(os.listdir(path))):
    j = 0
    shadow_imgs = []
    for i in latents:
        img = Image.fromarray(i).convert('RGB')
        width, height = img.width, img.height
        img = img.resize((size, size), Image.LANCZOS)
        img = img_transform(img)
        img = torch.unsqueeze(img, dim=0)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        G1.to(device)
        G1.eval()

        with torch.no_grad():
            detected_shadow = G1(img.to(device))
            detected_shadow = detected_shadow.to(torch.device('cpu'))
            detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
            detected_shadow = detected_shadow.resize((width, height), Image.LANCZOS)
            detected_shadow = np.array(detected_shadow)
            shadow_imgs.append(detected_shadow)
            # detected_shadow.save(out_path + '/detected_shadow_' + str(j) + '.jpg')
            j += 1

    np.save(out_path + '_shadow_images.npy', shadow_imgs)


def detect_shadows_img(path, out_path, size, img_transform, idx):
    print('Looking for shadows in generated images and selecting best one ...')
    G1 = Generator(input_channels=3, output_channels=1)

    G1_weights = torch.load('../models/st_cgan/ST-CGAN_G1_' + '1500.pth')
    G1.load_state_dict(fix_model_state_dict(G1_weights))

    # for image in tqdm(sorted(os.listdir(path))):
    img = Image.open(path)
    width, height = size, size
    img = img.resize((width, height), Image.LANCZOS)
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    G1.eval()

    with torch.no_grad():
        detected_shadow = G1(img.to(device))
        detected_shadow = detected_shadow.to(torch.device('cpu'))
        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow = detected_shadow.resize((width, height), Image.LANCZOS)
        detected_shadow = np.array(detected_shadow)
        detected_shadow = Image.fromarray(detected_shadow)
        # detected_shadow.save(out_path + '/' + idx[:-4] + '.jpg')
        detected_shadow.save(out_path + '/' + idx + '.jpg')



