import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import math
import cv2
from tqdm import tqdm
from PIL import Image, ImageStat
from torchvision import transforms

from utils.model_utils import setup_model
from utils.common import tensor2im, tensor2arr, tensor2im_no_transpose
from utils.face_parsing import evaluate
from utils.detect_shadows import detect_shadows
from datasets.shadow_detection_data_loader import ImageTransformOwn
from gan_control.inference.controller import Controller
from gan_control.utils.spherical_harmonics_utils import sh_eval_basis_1


def inspect_latents(img, args, only_original_img=False):
    dirname = os.getcwd()
    if only_original_img:
        device = "cuda"
        ckpt = dirname + '/checkpoints/stylegan2-ffhq-config-f.pt'
        latents = np.load(os.path.join(dirname, 'images/outputs/inversions/', img, 'original_latents.npy'),
                          allow_pickle=True)
        latents = torch.tensor(latents)
        latents = latents.view(1, 18, 512)
        save_dir = os.path.join(dirname, 'images/outputs/inversions/')

        imgs = []
        for v in tqdm(latents):  # take only first style vector
            latent_code = v.cpu().numpy()
            latent_vec = np.copy(latent_code)
            latent_vec = torch.tensor(latent_vec)
            imgs.append(latent_vec)

            break

        feed_forward(args, ckpt, device, os.path.join(dirname, save_dir), img, imgs, mode='original')
    else:
        device = "cuda"
        ckpt = os.path.join(dirname, 'checkpoints/stylegan2-ffhq-config-f.pt')
        latents = np.load(os.path.join(dirname, 'images/outputs/inversions/' + img + '/' + 'original_latents.npy'),
                          allow_pickle=True)
        latents = torch.tensor(latents)
        latents = latents.view(1, 18, 512)
        save_dir = os.path.join(dirname, 'images/outputs/inversions/')

        for v in tqdm(latents):  # take only first style vector
            latent_code = v.cpu().detach().numpy()
            imgs = []
            n_random_imgs = 400
            parameter_dict = {}
            value = 5
            for i in range(n_random_imgs):
                random_idx = []
                changes = []
                latent_vec = np.copy(latent_code)  # copy which will be manipulated
                for _ in range(3):  # 10 random indizes
                    random_idx.append((random.randrange(0, 15), random.randrange(0, 511)))  # random index

                for idx in random_idx:
                    if random.uniform(0, 1) < 0.5:  # increase / decrease value at position idx with probability 0.5
                        latent_vec[idx[0]][idx[1]] = latent_vec[idx[0]][idx[1]] + value
                        changes.append(value)
                    else:
                        latent_vec[idx[0]][idx[1]] = latent_vec[idx[0]][idx[1]] - value
                        changes.append(-value)

                parameter_dict['image_' + str(i) + '.jpg'] = (random_idx, changes)
                latent_vec = torch.tensor(latent_vec)
                imgs.append(latent_vec)
            np.save(os.path.join(save_dir, img, img, '_parameters.npy'), parameter_dict)
            print('Feeding ' + str(n_random_imgs) + ' random vectors into StyleGAN ...')

            feed_forward(args, ckpt, device, os.path.join(dirname, save_dir), img, imgs)

            break


def feed_forward(args, ckpt, device, save_dir, img, imgs, mode='reference'):
    dirname = os.getcwd()
    net, opts = setup_model(dirname + args.ckpt, device)
    generator = net.decoder
    generator.eval()
    print('Feeding into StyleGAN network ...')
    if mode == 'original':
        i = 0
        for image in imgs:
            out_img, _ = generator([image.cuda().unsqueeze(0)], None, input_is_latent=True, randomize_noise=False,
                                   return_latents=True)
            horizontal_concat_image = torch.cat(list(out_img), 2)
            final_img = tensor2im(horizontal_concat_image)
            final_img.save(save_dir + img + '/' + img + '_original_inversion/' + img + '_original_inversion.jpg', 'JPEG')
            i = i + 1
    elif mode == 'reference':
        i = 0
        image_latents = []
        for image in tqdm(imgs):
            out_img, _ = generator([image.cuda().unsqueeze(0)], None, input_is_latent=True, randomize_noise=False,
                                   return_latents=True)
            horizontal_concat_image = torch.cat(list(out_img), 2)
            final_img = tensor2arr(horizontal_concat_image)
            image_latents.append(final_img)

        np.save(save_dir + img + '/' + img + '_latent_images.npy', image_latents)
    else:
        pass


def calc_brightness(image, path='/images/outputs/inversions/'):
    scores = {}
    for img in sorted(os.listdir(path + image)):
        im = Image.open(path + image + '/' + img)
        stat = ImageStat.Stat(im)
        r, g, b = stat.mean
        scores[img] = math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

    scores_sorted = list(reversed(sorted(scores.items(), key=lambda item: item[1])[0:10]))

    # save best images
    j = 0
    for img in scores_sorted:
        open_img = Image.open(path + image + '/' + img[0])
        open_img.save(path + image + '/' + img + '_best_results' + '/test_' + str(j) + '.jpg')
        j += 1

    print(scores_sorted)


def find_brightest_image(image):
    dirname = os.getcwd()
    save_dir = os.path.join(dirname, 'images/outputs/inversions/' + image)

    detect_shadows(os.path.join(dirname, 'images/outputs/inversions/' + image + '/' + image),
                   os.path.join(dirname, 'images/outputs/inversions/' + image + '/' + image),
                   size=256, img_transform=ImageTransformOwn(size=286, mean=(0.5,), std=(0.5,)))

    dir_shadows = os.path.join(dirname, 'images/outputs/inversions/' + image + '/' + image + '_shadow_images.npy')
    dir_images = os.path.join(dirname, 'images/outputs/inversions/' + image + '/' + image + '_latent_images.npy')

    shadow_images = np.load(dir_shadows, allow_pickle=True)
    images = np.load(dir_images, allow_pickle=True)
    num_shadow = {}
    key = 0
    for img in shadow_images:
        white = (img > 100).sum()

        num_shadow[str(key)] = white
        key += 1

    res = list(sorted(num_shadow.items(), key=lambda item: item[1])[0:10])

    # save best image
    best_idx = res[0][0]
    best_img = images[int(best_idx)]

    plt.imsave(save_dir + '/' + image + '_final_result/' + image + '_' + best_idx + '.jpg', best_img)


def blur_background(img):
    dirname = os.getcwd()
    # blur background with newly generated facemask
    evaluate(dspth=os.path.join(dirname, '/images/outputs/inversions/' + img + '_final_result'),
             respth=os.path.join(dirname, '/images/outputs/inversions/' + img + '_face_mask'),
             cp=os.path.join(dirname, '/models/face_parsing/79999_iter.pth'), resize=False)

    # load mask, original image and final image
    mask_file = os.listdir(os.path.join(dirname, '/images/outputs/inversions/' + img + '_face_mask/')[0])
    mask = cv2.imread(os.path.join(dirname, '/images/outputs/inversions/' + img + '_face_mask/' + mask_file))
    image = cv2.imread(os.path.join(dirname, '/images/inputs/images/' + img + '.png'))
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    final_img_path = ""
    for f in os.listdir(os.path.join(dirname, '/images/outputs/inversions/' + img + '_final_result/')):
        final_img_path = os.path.join(dirname, '/images/outputs/inversions/' + img + '_final_result/' + f)

    final_img = cv2.imread(final_img_path)

    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    bg = np.copy(image)
    bg[(mask < 255).any(-1)] = [255, 255, 255]

    no_bg_img = cv2.addWeighted(bg, 1.0, image, 0.0, 0, bg)

    for x in range(no_bg_img.shape[0]):
        for y in range(no_bg_img.shape[1]):
            if (no_bg_img[x][y] == [255]).all():
                no_bg_img[x][y] = final_img[x][y]

    no_bg_img = cv2.cvtColor(no_bg_img, cv2.COLOR_BGR2RGB)

    plt.imsave('test.png', no_bg_img)


def illuminate_img(image):
    dirname = os.getcwd()
    save_dir = os.path.join(dirname, 'images/outputs/inversions/' + image)

    controller_path = dirname + '/resources/gan_models/controller_age015id025exp02hai04ori02gam15'
    controller = Controller(controller_path)

    # load latents
    latents = np.load(os.path.join(dirname, 'images/outputs/inversions/', image, 'original_latents.npy'),
                      allow_pickle=True)
    latents = torch.tensor(latents)
    latents = latents.view(1, 18, 512)
    latents = latents[0]

    # Strong front illumination
    strangth = 0.7
    illumination_control = torch.tensor([sh_eval_basis_1(0, 0, 1)]) * strangth
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=latents,
                                                                           input_is_latent=True,
                                                                           gamma=illumination_control)

    print(image_tensors.size())
    grid_img = controller.make_resized_grid_image(image_tensors, resize=480, nrow=8)
    grid_img.save(save_dir + '/' + image + '_final_result/' + image + '_test.jpg')
    final_img = image_tensors[0]
    final_img = final_img.permute(1, 2, 0)
    final_img = tensor2im_no_transpose(final_img)
    plt.imsave(save_dir + '/' + image + '_final_result/' + image + '.jpg', final_img)


def illuminate_projected_img(image_tensor, image_name):
    dirname = os.getcwd()
    save_dir = os.path.join(dirname, 'images/outputs/inversions/' + image_name)

    controller_path = dirname + '/resources/gan_models/controller_age015id025exp02hai04ori02gam15'
    controller = Controller(controller_path)
    # Strong front illumination
    strangth = 0.7
    illumination_control = torch.tensor([sh_eval_basis_1(0, 0, 1)]) * strangth
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=image_tensor,
                                                                           input_is_latent=True,
                                                                           gamma=illumination_control)

    print(image_tensors.size())
    grid_img = controller.make_resized_grid_image(image_tensors, resize=512, nrow=8)
    grid_img.save(save_dir + '/' + image_name + '_final_result/' + image_name + '_test.jpg')