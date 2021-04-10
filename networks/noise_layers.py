import re
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def do_resize(resize_command):
    matches = re.match(r'\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))


def do_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def do_crop(crop_command):
    matches = re.match(r'\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = do_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))


def do_cropout(cropout_command):
    matches = re.match(r'\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = do_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def do_dropout(dropout_command):
    matches = re.match(r'\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """

    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method

    def forward(self, noised_and_cover):
        resize_ratio = np.random.rand() * (self.resize_ratio_max - self.resize_ratio_min) + self.resize_ratio_min
        noised_image = noised_and_cover[0]
        noised_and_cover[0] = F.interpolate(
            noised_image,
            scale_factor=(resize_ratio, resize_ratio),
            mode=self.interpolation_method,
            recompute_scale_factor=True
        )

        return noised_and_cover


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(
        (np.random.rand() * (height_ratio_range[1] - height_ratio_range[0]) + height_ratio_range[0]) * image_height))
    # may have a bug
    remaining_width = int(
        np.rint(
            (np.random.rand() * (width_ratio_range[1] - width_ratio_range[0]) + width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """

    def __init__(self, height_ratio_range, width_ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range,
                                                                     self.width_ratio_range)

        noised_and_cover[0] = noised_image[
                              :,
                              :,
                              h_start: h_end,
                              w_start: w_end].clone()

        return noised_and_cover


class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """

    def __init__(self, height_ratio_range, width_ratio_range):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        cropout_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        noised_and_cover[0] = noised_image * cropout_mask + cover_image * (torch.ones_like(cropout_mask) - cropout_mask)
        return noised_and_cover


class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """

    def __init__(self, keep_ratio_range):
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        mask_percent = np.random.uniform(self.keep_min, self.keep_max)

        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (torch.ones_like(mask_tensor) - mask_tensor)
        return [noised_image, cover_image]


class Identity(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, noised_and_cover):
        return noised_and_cover


class JpegCompression(nn.Module):
    def __init__(self, device, yuv_keep_weights=(25, 9, 9)):
        super(JpegCompression, self).__init__()
        self.device = device

        self.dct_conv_weights = torch.tensor(self.gen_filters(8, 8, self.dct_coeff), dtype=torch.float32).to(
            self.device)
        self.dct_conv_weights.unsqueeze_(1)
        self.idct_conv_weights = torch.tensor(self.gen_filters(8, 8, self.idct_coeff), dtype=torch.float32).to(
            self.device)
        self.idct_conv_weights.unsqueeze_(1)

        self.yuv_keep_weighs = yuv_keep_weights
        self.keep_coeff_masks = []

        self.jpeg_mask = None

        # create a new large mask which we can use by slicing for images which are smaller
        self.create_mask((1000, 1000))

    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weighs):
                mask = torch.from_numpy(self.get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_mask(self, image_shape):
        if self.jpeg_mask.shape < image_shape:
            self.create_mask(image_shape)
        # return the correct slice of it
        return self.jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()

    def apply_conv(self, image, filter_type: str):

        if filter_type == 'dct':
            filters = self.dct_conv_weights
        elif filter_type == 'idct':
            filters = self.idct_conv_weights
        else:
            raise ValueError('Unknown filter_type value.')

        image_conv_channels = []
        for channel in range(image.shape[1]):
            image_yuv_ch = image[:, channel, :, :].unsqueeze_(1)
            image_conv = F.conv2d(image_yuv_ch, filters, stride=8)
            image_conv = image_conv.permute(0, 2, 3, 1)
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], 8, 8)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                      image_conv.shape[1] * image_conv.shape[2],
                                                      image_conv.shape[3] * image_conv.shape[4])

            image_conv.unsqueeze_(1)

            image_conv_channels.append(image_conv)

        image_conv_stacked = torch.cat(image_conv_channels, dim=1)

        return image_conv_stacked

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        # pad the image so that we can do dct on 8x8 blocks
        pad_height = (8 - noised_image.shape[2] % 8) % 8
        pad_width = (8 - noised_image.shape[3] % 8) % 8

        noised_image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(noised_image)

        # convert to yuv
        image_yuv = torch.empty_like(noised_image)
        self.rgb2yuv(noised_image, image_yuv)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        # apply dct
        image_dct = self.apply_conv(image_yuv, 'dct')
        # get the jpeg-compression mask
        mask = self.get_mask(image_dct.shape[1:])
        # multiply the dct-ed image with the mask.
        image_dct_mask = torch.mul(image_dct, mask)

        # apply inverse dct (idct)
        image_idct = self.apply_conv(image_dct_mask, 'idct')
        # transform from yuv to to rgb
        image_ret_padded = torch.empty_like(image_dct)
        self.yuv2rgb(image_idct, image_ret_padded)

        # un-pad
        noised_and_cover[0] = \
            image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
            :image_ret_padded.shape[3] - pad_width].clone()

        return noised_and_cover

    def gen_filters(self, size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
        tile_size_x = 8
        filters = np.zeros((size_x * size_y, size_x, size_y))
        for k_y in range(size_y):
            for k_x in range(size_x):
                for n_y in range(size_y):
                    for n_x in range(size_x):
                        filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y,
                                                                                     size_y) * dct_or_idct_fun(
                            n_x,
                            k_x,
                            size_x)
        return filters

    def rgb2yuv(self, image_rgb, image_yuv_out):
        """ Transform the image from rgb to yuv """
        image_yuv_out[:, 0, :, :] = \
            0.299 * image_rgb[:, 0, :, :].clone() + \
            0.587 * image_rgb[:, 1, :, :].clone() + \
            0.114 * image_rgb[:, 2, :, :].clone()
        image_yuv_out[:, 1, :, :] = \
            -0.14713 * image_rgb[:, 0, :, :].clone() + \
            -0.28886 * image_rgb[:, 1, :, :].clone() + \
            0.436 * image_rgb[:, 2, :, :].clone()
        image_yuv_out[:, 2, :, :] = \
            0.615 * image_rgb[:, 0, :, :].clone() + \
            -0.51499 * image_rgb[:, 1, :, :].clone() \
            + -0.10001 * image_rgb[:, 2, :, :].clone()

    def yuv2rgb(self, image_yuv, image_rgb_out):
        """ Transform the image from yuv to rgb """
        image_rgb_out[:, 0, :, :] = \
            image_yuv[:, 0, :, :].clone() + \
            1.13983 * image_yuv[:, 2, :, :].clone()
        image_rgb_out[:, 1, :, :] = \
            image_yuv[:, 0, :, :].clone() + \
            -0.39465 * image_yuv[:, 1, :, :].clone() + \
            -0.58060 * image_yuv[:, 2, :, :].clone()
        image_rgb_out[:, 2, :, :] = \
            image_yuv[:, 0, :, :].clone() + \
            2.03211 * image_yuv[:, 1, :, :].clone()

    def get_jpeg_yuv_filter_mask(self, image_shape: tuple, window_size: int, keep_count: int):
        mask = np.zeros((window_size, window_size), dtype=np.uint8)

        index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                             key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

        for i, j in index_order[0:keep_count]:
            mask[i, j] = 1

        return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                              int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]

    def dct_coeff(self, n, k, n_):
        return np.cos(np.pi / n_ * (n + 1. / 2.) * k)

    def idct_coeff(self, n, k, n_):
        return (int(0 == n) * (- 1 / 2) + np.cos(
            np.pi / n_ * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * n_))


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.min_value = 0.0
        self.max_value = 255.0
        self.N = 10
        self.weights = torch.tensor([((-1) ** (n + 1)) / (np.pi * (n + 1)) for n in range(self.N)]).to(device)
        self.scales = torch.tensor([2 * np.pi * (n + 1) for n in range(self.N)]).to(device)
        for _ in range(4):
            self.weights.unsqueeze_(-1)
            self.scales.unsqueeze_(-1)

    def fourier_rounding(self, tensor):
        z = torch.mul(self.weights, torch.sin(torch.mul(tensor, self.scales)))
        z = torch.sum(z, dim=0)
        return tensor + z

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        noised_image = self.transform(noised_image, (0, 255))
        noised_image = self.fourier_rounding(noised_image.clamp(self.min_value, self.max_value))
        noised_image = self.transform(noised_image, (noised_and_cover[0].min(), noised_and_cover[0].max()))
        return [noised_image, noised_and_cover[1]]

    def transform(self, tensor, target_range):
        source_min = tensor.min()
        source_max = tensor.max()

        # normalize to [0, 1]
        tensor_target = (tensor - source_min) / (source_max - source_min)
        # move to target range
        tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
        return tensor_target
