import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from networks.HidDeN import Encoder, Decoder, Noiser
from utils.utils import exec_val
from utils.metric import cal_single_psnr


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_dir', type=str, default='datasets/samples')
    parser.add_argument('--result_dir', type=str, default='datasets/results')
    parser.add_argument('--epochs', type=str, default=30)

    parser.add_argument('--input_size', type=tuple, default=(64, 64))
    parser.add_argument('--info_size', type=int, default=30)

    parser.add_argument('--use_noise', type=bool, default=False)

    args = parser.parse_args()
    return args


def encode_decode(args, logger_name):
    # Network
    if args.use_noise is True:
        noise_layers = {
            'crop': '((0.4,0.55),(0.4,0.55))',
            'cropout': '((0.25,0.35),(0.25,0.35))',
            'dropout': '(0.25,0.35)',
            'jpeg': '()',
            'resize': '(0.4,0.6)',
        }  # This is a combined noise used in the paper
    else:
        noise_layers = dict()
    input_height, input_width = args.input_size
    encoder = Encoder(input_height, input_width, args.info_size)
    noiser = Noiser(noise_layers, torch.device('cpu'))
    decoder = Decoder(args.info_size)
    encoder.cpu()
    noiser.cpu()
    decoder.cpu()

    epoch = args.epochs
    enc_param = torch.load('ckpt/{}/models/encoder-epoch{:0>3d}.pth'.format(logger_name, epoch))
    if args.use_noise:
        noiser_param = torch.load('ckpt/{}/models/noiser-epoch{:0>3d}.pth'.format(logger_name, epoch))
    else:
        noiser_param = None
    dec_param = torch.load('ckpt/{}/models/decoder-epoch{:0>3d}.pth'.format(logger_name, epoch))

    encoder.load_state_dict(enc_param)
    if args.use_noise:
        noiser.load_state_dict(noiser_param)
    decoder.load_state_dict(dec_param)

    os.makedirs(args.result_dir, exist_ok=True)
    files_list = os.listdir(args.sample_dir)
    metric = dict()
    for file in files_list:
        img_path = os.path.join(args.sample_dir, file)
        dst_path = os.path.join(args.result_dir, file)
        metric[file] = dict()
        print('Processing image {}'.format(img_path))

        ori_image = cv2.imread(img_path)  # H, W, C
        image_height = ori_image.shape[0]
        image_width = ori_image.shape[1]
        image = Image.fromarray(cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB))
        transforms.CenterCrop(args.input_size),
        data_trans = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = data_trans(image)  # C, H, W

        image = torch.unsqueeze(image, dim=0)

        image = image.cpu()
        info = torch.randint(0, 2, size=(1, args.info_size)).to(torch.device('cpu'), dtype=torch.float32)

        encoder.eval()
        noiser.eval()
        decoder.eval()

        encoded_image = encoder(image, info)
        noised_and_cover = noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_info = decoder(noised_image)

        decoded_rounded = decoded_info.detach().cpu().numpy().round().clip(0, 1)
        bitwise_sum_err = np.sum(np.abs(decoded_rounded - info.detach().cpu().numpy()))
        bitwise_avg_err = bitwise_sum_err / (1 * info.shape[1])

        info = "".join([str(int(i)) for i in info.tolist()[0]])
        decoded_rounded = "".join([str(int(i)) for i in decoded_rounded.tolist()[0]])

        print("info:        ", info)
        print("decoded_info:", decoded_rounded)

        exec_val(image, encoded_image, dst_path, resize_to=(image_height, image_width), mode='single')

        encoded_image_ = cv2.imread(dst_path)

        metric[file]['err'] = round(bitwise_avg_err, 3)
        metric[file]['psnr'] = round(cal_single_psnr(ori_image, encoded_image_), 3)

    for k, v in sorted(metric.items()):
        print('{: <25}:{}'.format(k, v))


if __name__ == '__main__':
    args = parse()
    encode_decode(args, logger_name='train_21-04-09_13-06-53')
