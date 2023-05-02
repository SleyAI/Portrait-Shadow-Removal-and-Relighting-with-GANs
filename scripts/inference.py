import os
import shutil
import argparse
import numpy as np
import torch
import cv2
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.model_utils import setup_e4e_model
from scripts.edit_latents import inspect_latents, find_brightest_image
from utils.face_parsing import evaluate


def run(args):
    start = datetime.now()
    os.chdir('../')
    dirname = os.getcwd()
    save_dir = os.path.join(dirname, 'images/outputs/inversions/')
    removed_bg_images_dir = os.path.join(dirname, 'images/inputs/removed_background')
    face_mask_dir = os.path.join(dirname, 'images/inputs/face_masks/')
    input_dir = os.path.join(dirname, 'images/inputs/images/')

    # Clear inputs from earlier experiments
    for filename in os.listdir(face_mask_dir):
        file_path = os.path.join(face_mask_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    for filename in os.listdir(removed_bg_images_dir):
        file_path = os.path.join(removed_bg_images_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print('Generating Face Masks...')
    evaluate(dspth=os.path.join(dirname, 'images/inputs/images'),
             respth=face_mask_dir,
             cp=os.path.join(dirname, 'models/face_parsing/79999_iter.pth'), resize=True)

    print('Removing Background...')
    # remove background via face mask
    images = []
    for i, m in tqdm(zip(sorted(os.listdir(input_dir)), sorted(os.listdir(face_mask_dir)))):
        image = i[:-4]
        images.append(image)
        mask = cv2.imread(face_mask_dir + '/' + m)
        face = cv2.imread(input_dir + '/' + i)
        face = cv2.resize(face, (512, 512))

        _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
        # copy where we'll assign the new values
        no_bg = np.copy(face)
        # boolean indexing and assignment based on mask
        no_bg[(mask == 255).all(-1)] = [255, 255, 255]

        no_bg_img = cv2.addWeighted(no_bg, 1.0, face, 0.0, 0, no_bg)
        cv2.imwrite(dirname + '/images/inputs/removed_background/' + i, no_bg_img)

    for i in images:
        if not os.path.exists(save_dir + i):
            os.mkdir(save_dir + i)

    for i in images:
        if not os.path.exists(save_dir + i + '/' + i + '_final_result'):
            os.mkdir(save_dir + i + '/' + i + '_final_result')
        if not os.path.exists(save_dir + i + '/' + i + '_original_inversion'):
            os.mkdir(save_dir + i + '/' + i + '_original_inversion')
        if not os.path.exists(save_dir + i + '/' + i):
            os.mkdir(save_dir + i + '/' + i)

        shutil.copyfile(input_dir + i + '.png',
                        save_dir + i + '/' + i + '/' + i + '.png')

    for i in images:
        if not os.path.exists(save_dir + i):
            os.mkdir(save_dir + i)

    # update test options with options used during training
    model_path = dirname + args.ckpt
    net, opts = setup_e4e_model(model_path, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = dirname + args.images_dir
    test_dataset = InferenceDataset(root=images_path + '/removed_background',
                                    transform=transforms_dict['transform_test'],
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    if args.n_sample is None:
        args.n_sample = len(test_dataset)

    print('Beginning with GAN Inversion of original image ...')
    # initial inversion
    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
    # perform high-fidelity inversion or editing
    for i, batch in tqdm(enumerate(data_loader)):
        if args.n_sample is not None and i > args.n_sample:
            print('inference finished!')
            break

        # calculate the distortion map
        imgs, _ = generator([latent_codes[i].unsqueeze(0).to(device)], None, input_is_latent=True,
                            randomize_noise=False, return_latents=True)

        edit_latents = latent_codes[i].unsqueeze(0).to(device)

        # consultation fusion
        latents = edit_latents.cpu().detach().numpy()
        np.save(os.path.join(save_dir + images[i], 'original_latents.npy'), latents)

    print('GAN Inversion of original image successful')

    for i in images:
        inspect_latents(i, args, only_original_img=True)
    # initial inversion
    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

    # perform high-fidelity inversion or editing
    for i, (j, batch) in zip(images, enumerate(data_loader)):
        if args.n_sample is not None and j > args.n_sample:
            print('inference finished!')
            break

        # calculate the distortion map
        imgs, _ = generator([latent_codes[j].unsqueeze(0).to(device)], None, input_is_latent=True,
                            randomize_noise=False, return_latents=True)

        edit_latents = latent_codes[j].unsqueeze(0).to(device)

        # consultation fusion
        latents = edit_latents.cpu().detach().numpy()
        np.save(os.path.join(save_dir + i, 'latents.npy'), latents)

    print('GAN Inversion successful')

    # Latent Vector Editing
    print('Beginning with editing...')
    for i in images:
        inspect_latents(i, args, only_original_img=False)
        find_brightest_image(i)

        # delete latents .npy files
        path = save_dir + i
        for item in os.listdir(path):
            if item.endswith(".npy"):
                if not 'parameters' in item:
                    os.remove(os.path.join(path, item))
    print('Total time: ' + str(datetime.now() - start))
    print('Average time per image: ' + str((datetime.now() - start)/len(images)))

def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dirname = os.getcwd()
    images_path = dirname + args.images_dir
    print(f"images path: {images_path}")
    test_dataset = InferenceDataset(root=images_path + '/images',
                                    transform=transforms_dict['transform_test'],
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    run(args)