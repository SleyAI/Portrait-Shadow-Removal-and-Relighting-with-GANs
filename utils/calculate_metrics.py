import lpips
import cv2
import torch
import numpy as np

from image_similarity_measures.quality_metrics import rmse, ssim, fsim, psnr
from PIL import ImageChops


def calc_lpips(a, b):
    # 0 similar, 1.0 different
    a = torch.FloatTensor(a / 255.0)
    b = torch.FloatTensor(b / 255.0)

    a = a.view(1, 3, 256, 256)
    b = b.view(1, 3, 256, 256)

    loss_fn_alex = lpips.LPIPS(net='alex')
    distance = loss_fn_alex(a, b)

    return distance.item()


def calc_ssim(a, b):
    # 1.0 identical, 0 different
    distance = ssim(a, b)

    return distance


def calc_fsim(a, b):
    # 1.0 identical, 0 different
    distance = fsim(a, b)

    return distance


def calc_psnr(a, b):
    distance = psnr(a, b)

    return distance


def calc_rmse(a, b):
    distance = rmse(a, b)

    return distance


def calc_metrics(dir_a, dir_b):
    distances = {}

    # a = cv2.imread(dir_a)
    # b = cv2.imread(dir_b)

    # print(dir_a)
    # print(dir_b)

    a = cv2.resize(cv2.imread(dir_a), (256, 256))
    b = cv2.resize(cv2.imread(dir_b), (256, 256))

    distances['LPIPS'] = calc_lpips(a, b)
    distances['SSIM'] = calc_ssim(a, b)
    distances['FSIM'] = calc_fsim(a, b)

    return distances

