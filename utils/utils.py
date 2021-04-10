import torch
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid


def exec_val(original_images, encoded_images, file_name, resize_to=None, mode='multiple'):
    original_images = original_images[:original_images.shape[0], :, :, :]
    encoded_images = encoded_images[:encoded_images.shape[0], :, :, :]

    original_images = (original_images + 1) / 2
    encoded_images = (encoded_images + 1) / 2

    if resize_to is not None:
        original_images = F.interpolate(original_images, size=resize_to)
        encoded_images = F.interpolate(encoded_images, size=resize_to)

    if mode == 'multiple':
        stacked_images = torch.cat([original_images, encoded_images], dim=0)
        torchvision.utils.save_image(stacked_images, file_name, original_images.shape[0])
        stacked_images = make_grid(stacked_images, nrow=20)
        return stacked_images
    elif mode == 'single':
        torchvision.utils.save_image(encoded_images, file_name, original_images.shape[0])
    else:
        raise ValueError('Invalid mode parameter {}'.format(mode))
