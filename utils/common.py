from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Log images
def log_input_image(x, opts):
    if opts.label_nc == 0:
        return tensor2im(x)
    elif opts.label_nc == 1:
        return tensor2sketch(x)
    else:
        return tensor2map(x)


def tensor2map(var):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)


def tensor2sketch(var):
    im = var[0].cpu().detach().numpy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = (im * 255).astype(np.uint8)
    return Image.fromarray(im)


# Visualization utils
def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
            [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
            [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def tensor2im_no_transpose(var):
    var = var.cpu().detach().numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    var = var.astype(np.uint8)
    return var


def tensor2arr(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')


def tensor2arr_no_transpose(var):
    var = var.cpu().detach().numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    n_outputs = len(log_hooks[0]['output_face']) if type(log_hooks[0]['output_face']) == list else 1
    fig = plt.figure(figsize=(8 + (n_outputs * 2), 3 * display_count))
    gs = fig.add_gridspec(display_count, (3 + n_outputs))
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        vis_faces_iterative(hooks_dict, fig, gs, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_iterative(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['w_inversion'])
    plt.title('W-Inversion\n')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']), float(hooks_dict['diff_target'])))
    for idx, output_idx in enumerate(range(len(hooks_dict['output_face']) - 1, -1, -1)):
        output_image, similarity = hooks_dict['output_face'][output_idx]
        fig.add_subplot(gs[i, 3 + idx])
        plt.imshow(output_image)
        plt.title('Output {}\n Target Sim={:.2f}'.format(output_idx, float(similarity)))
