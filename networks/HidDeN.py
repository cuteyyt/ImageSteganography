import torch
import numpy as np
import torch.nn as nn
from networks.noise_layers import Identity, JpegCompression, Quantization, do_resize, do_crop, do_cropout, do_dropout


def conv_bn_relu_layers(channels_in, channels_out, stride=1):
    """
    Building block used in HiDDeN network.
    Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    layers = nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True)
    )
    return layers


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, input_height, input_width, info_size):
        super(Encoder, self).__init__()
        self.encoder_channels = 64
        self.encoder_blocks = 4

        self.input_height = input_height
        self.input_width = input_width
        self.info_size = info_size
        self.conv_channels = self.encoder_channels
        self.num_blocks = self.encoder_blocks

        layers = [conv_bn_relu_layers(3, self.conv_channels)]

        for _ in range(self.encoder_blocks - 1):
            layer = conv_bn_relu_layers(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = conv_bn_relu_layers(self.conv_channels + 3 + self.info_size,
                                                      self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, info):
        # First, add two dummy dimensions in the end of the info.
        # This is required for the .expand to work correctly
        expanded_info = info.unsqueeze(-1)
        expanded_info.unsqueeze_(-1)

        expanded_info = expanded_info.expand(-1, -1, self.input_height, self.input_width)
        encoded_image = self.conv_layers(image)
        # concatenate expanded info and image
        # print(encoded_image.cpu().dtype)
        concat = torch.cat([expanded_info, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, info_size):
        super(Decoder, self).__init__()
        self.decoder_channels = 64
        self.decoder_blocks = 7
        self.channels = self.decoder_channels
        self.info_size = info_size

        layers = [conv_bn_relu_layers(3, self.channels)]
        for _ in range(self.decoder_blocks - 1):
            layers.append(conv_bn_relu_layers(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(conv_bn_relu_layers(self.channels, self.info_size))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(self.info_size, self.info_size)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network.
    Receives an image and has to figure out whether it has info inserted into it, or not.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_channels = 64
        self.discriminator_blocks = 3

        layers = [conv_bn_relu_layers(3, self.discriminator_channels)]
        for _ in range(self.discriminator_blocks - 1):
            layers.append(conv_bn_relu_layers(self.discriminator_channels, self.discriminator_channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(self.discriminator_channels, 1)

    def forward(self, image):
        x = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        # x = torch.sigmoid(X)
        return x


basic_noise = ['resize', 'crop', 'cropout', 'dropout', 'jpeg', 'quant', 'identity']


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module.
    """

    def __init__(self, noise_layers, device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        for k, v in noise_layers.items():
            if k not in basic_noise:
                raise ValueError('Wrong noise layer type {}'.format(k))
            elif k == 'resize':
                self.noise_layers.append(do_resize(v))
            elif k == 'crop':
                self.noise_layers.append(do_crop(v))
            elif k == 'cropout':
                self.noise_layers.append(do_cropout(v))
            elif k == 'dropout':
                self.noise_layers.append(do_dropout(v))
            elif k == 'jpeg':
                self.noise_layers.append(JpegCompression(device))
            elif k == 'quant':
                self.noise_layers.append(Quantization())
            elif k == 'identity':
                pass

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        # print(random_noise_layer)
        return random_noise_layer(encoded_and_cover)
