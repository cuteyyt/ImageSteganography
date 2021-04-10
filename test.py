import os
import time
import torch
import argparse
import numpy as np

from data.DataLoader import get_data_loaders
from networks.HidDeN import Encoder, Decoder, Noiser
from utils.logger import Logger
from utils.utils import exec_val


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--trainset_dir', type=str, default='datasets/celeba_resize/train')
    parser.add_argument('--valset_dir', type=str, default='datasets/celeba_resize/test')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--validate_batch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--input_size', type=tuple, default=(64, 64))
    parser.add_argument('--info_size', type=int, default=30)

    parser.add_argument('--relative_loss', type=bool, default=False)
    parser.add_argument('--use_noise', type=bool, default=False)

    args = parser.parse_args()
    return args


def test(args, logger_name=''):
    # Configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_height, input_width = args.input_size

    date = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    logger = Logger(log_root='logs/', logger_name=logger_name + '_' + date)

    for k, v in args.__dict__.items():
        logger.add_text('configuration', "{}: {}".format(k, v))

    # Dataset
    _, test_loader = get_data_loaders(args)
    print("test set size:", len(test_loader.dataset))

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
    encoder = Encoder(input_height, input_width, args.info_size)
    noiser = Noiser(noise_layers, torch.device('cuda'))
    decoder = Decoder(args.info_size)
    encoder.cuda()
    noiser.cuda()
    decoder.cuda()

    # Testing
    dir_save = 'ckpt/{}'.format(logger.log_name)
    os.makedirs(dir_save, exist_ok=True)

    global_step = 1
    for epoch in range(1, args.epochs + 1):
        bitwise_err = 0.
        step = 1
        img_dir = os.path.join(dir_save, 'epoch{:0>3d}'.format(epoch))
        os.makedirs(img_dir, exist_ok=True)
        for image, _ in test_loader:
            image = image.cuda()
            batch_size = image.shape[0]
            info = torch.randint(0, 2, size=(batch_size, args.info_size)).to(device, dtype=torch.float32)

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

            encoder.eval()
            noiser.eval()
            decoder.eval()

            encoded_image = encoder(image, info)
            noised_and_cover = noiser([encoded_image, image])
            noised_image = noised_and_cover[0]
            decoded_info = decoder(noised_image)

            decoded_rounded = decoded_info.detach().cpu().numpy().round().clip(0, 1)
            bitwise_sum_err = np.sum(np.abs(decoded_rounded - info.detach().cpu().numpy()))
            bitwise_avg_err = bitwise_sum_err / (batch_size * info.shape[1])
            bitwise_err += bitwise_avg_err

            stack_image = exec_val(image, encoded_image,
                                   os.path.join(img_dir, '{:d}.png'.format(step)))

            logger.add_scalar('err_ratio', bitwise_avg_err, global_step)
            logger.add_image('image', stack_image, global_step)

            step += 1
            global_step += 1

        bitwise_err /= len(test_loader.dataset)
        logger.add_scalar('err_ratio_epoch', bitwise_err, epoch)


if __name__ == '__main__':
    args = parse()
    test(args, logger_name='train_21-04-09_13-06-53')
