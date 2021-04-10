import os
import math
import torch
import argparse
import numpy as np
import random
from collections import defaultdict
from torch.nn import functional as F

from data.DataLoader import get_data_loaders
from networks.HidDeN import Encoder, Decoder, Discriminator, Noiser
from utils.metric import AverageLoss
from utils.logger import Logger
from utils.utils import exec_val


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logger_name', type=str, default='train')
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--trainset_dir', type=str, default='datasets/celeba_resize/train')
    parser.add_argument('--valset_dir', type=str, default='datasets/celeba_resize/test')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--validate_batch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--input_size', type=tuple, default=(64, 64))
    parser.add_argument('--info_size', type=int, default=30)

    parser.add_argument('--adversarial_loss_constant', type=float, default=1e-3)
    parser.add_argument('--encoder_loss_constant', type=float, default=0.7)
    parser.add_argument('--decoder_loss_constant', type=float, default=1.)

    parser.add_argument('--relative_loss', type=bool, default=False)
    parser.add_argument('--use_noise', type=bool, default=False)

    args = parser.parse_args()
    return args


def train(args):
    # Configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_height, input_width = args.input_size

    logger = Logger(log_root='logs/', name=args.logger_name)

    for k, v in args.__dict__.items():
        logger.add_text('configuration', "{}: {}".format(k, v))

    # Dataset
    train_loader, val_loader = get_data_loaders(args)
    batchs_in_val = math.ceil(len(val_loader.dataset) / args.validate_batch)
    print("Train set size:", len(train_loader.dataset))
    print("Val set size:", len(val_loader.dataset))

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
    discriminator = Discriminator()
    encoder.cuda()
    noiser.cuda()
    decoder.cuda()
    discriminator.cuda()

    # Optimizers
    optimizer_enc = torch.optim.Adam(encoder.parameters())
    optimizer_dec = torch.optim.Adam(decoder.parameters())
    optimizer_dis = torch.optim.Adam(discriminator.parameters())

    # Training
    dir_save = 'ckpt/{}'.format(logger.log_name)
    os.makedirs(dir_save, exist_ok=True)
    os.makedirs(dir_save + '/images/', exist_ok=True)
    os.makedirs(dir_save + '/models/', exist_ok=True)
    training_losses = defaultdict(AverageLoss)

    info_fix = torch.randint(0, 2, size=(100, args.info_size)).to(device, dtype=torch.float32)
    image_fix = None
    for image, _ in val_loader:
        image_fix = image.cuda()  # 100 images for validate, the first batch
        break
    global_step = 1
    for epoch in range(1, args.epochs + 1):

        # Train one epoch
        for image, _ in train_loader:
            image = image.cuda()
            batch_size = image.shape[0]
            info = torch.randint(0, 2, size=(batch_size, args.info_size)).to(device, dtype=torch.float32)

            encoder.train()
            noiser.train()
            decoder.train()
            discriminator.train()
            # ---------------- Train the discriminator -----------------------------
            optimizer_dis.zero_grad()
            # train on cover
            y_real = torch.ones(batch_size, 1).cuda()
            y_fake = torch.zeros(batch_size, 1).cuda()

            d_on_cover = discriminator(image)
            encoded_image = encoder(image, info)
            d_on_encoded = discriminator(encoded_image.detach())

            if args.relative_loss:
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover - torch.mean(d_on_encoded),
                                                                     y_real)
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded - torch.mean(d_on_cover),
                                                                       y_fake)
                d_loss = d_loss_on_cover + d_loss_on_encoded
            else:
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, y_real)
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, y_fake)

                d_loss = d_loss_on_cover + d_loss_on_encoded

            d_loss.backward()
            optimizer_dis.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            d_on_cover = discriminator(image)
            encoded_image = encoder(image, info)
            noised_and_cover = noiser([encoded_image, image])
            noised_image = noised_and_cover[0]
            decoded_info = decoder(noised_image)
            d_on_encoded = discriminator(encoded_image)
            if args.relative_loss:
                g_loss_adv = \
                    (F.binary_cross_entropy_with_logits(d_on_encoded - torch.mean(d_on_cover), y_real) +
                     F.binary_cross_entropy_with_logits(d_on_cover - torch.mean(d_on_encoded), y_fake)) * 0.5
                g_loss_enc = F.mse_loss(encoded_image, image)
                g_loss_dec = F.mse_loss(decoded_info, info)
            else:
                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded, y_real)
                g_loss_enc = F.mse_loss(encoded_image, image)
                g_loss_dec = F.mse_loss(decoded_info, info)
            g_loss = args.adversarial_loss_constant * g_loss_adv + \
                     args.encoder_loss_constant * g_loss_enc + \
                     args.decoder_loss_constant * g_loss_dec

            g_loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            decoded_rounded = decoded_info.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err = \
                np.sum(np.abs(decoded_rounded - info.detach().cpu().numpy())) / \
                (batch_size * info.shape[1])

            losses = {
                'g_loss': g_loss.item(),
                'g_loss_enc': g_loss_enc.item(),
                'g_loss_dec': g_loss_dec.item(),
                'bitwise_avg_error': bitwise_avg_err,
                'g_loss_adv': g_loss_adv.item(),
                'd_loss_on_cover': d_loss_on_cover.item(),
                'd_loss_on_encoded': d_loss_on_encoded.item(),
                'd_loss': d_loss.item()
            }
            if logger:
                for name, loss in losses.items():
                    logger.add_scalar(name + '_iter', loss, global_step)
                    training_losses[name].update(loss)
            global_step += 1

        if logger:
            logger.add_scalar('d_loss_epoch', training_losses['d_loss'].avg, epoch)
            logger.add_scalar('g_loss_epoch', training_losses['g_loss'].avg, epoch)

        # Validate each epoch
        info_random = torch.randint(0, 2, size=(100, args.info_size)).to(device, dtype=torch.float32)
        image_random = None
        choice = random.randint(0, batchs_in_val - 2)
        # print(choice)
        for i, (image, _) in enumerate(val_loader):
            if i < choice:
                continue
            if image.shape[0] < 100:
                continue
            image_random = image.cuda()  # Grub the first batch
            break

        encoder.eval()
        noiser.eval()
        decoder.eval()
        discriminator.eval()

        encoded_image_random = encoder(image_random, info_random)
        noised_and_cover_random = noiser([encoded_image_random, image_random])
        noised_image_random = noised_and_cover_random[0]
        decoded_info_random = decoder(noised_image_random)

        encoded_image_fix = encoder(image_fix, info_fix)
        noised_and_cover_fix = noiser([encoded_image_fix, image_fix])
        noised_image_fix = noised_and_cover_fix[0]
        decoded_info_fix = decoder(noised_image_fix)

        decoded_rounded_fix = decoded_info_fix.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err_fix = \
            np.sum(np.abs(decoded_rounded_fix - info_fix.detach().cpu().numpy())) / \
            (100 * info_fix.shape[1])

        decoded_rounded_random = decoded_info_random.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err_random = \
            np.sum(np.abs(decoded_rounded_random - info_random.detach().cpu().numpy())) / \
            (100 * info_random.shape[1])

        stack_image_random = exec_val(image_random, encoded_image_random,
                                      os.path.join(dir_save, 'images', 'random_epoch{:0>3d}.png'.format(epoch)))
        stack_image_fix = exec_val(image_fix, encoded_image_fix,
                                   os.path.join(dir_save, 'images', 'fix_epoch{:0>3d}.png'.format(epoch)))
        if logger:
            logger.add_scalar('fix_err_ratio', bitwise_avg_err_fix, epoch)
            logger.add_scalar('random_err_ratio', bitwise_avg_err_random, epoch)
            logger.add_image('image_rand', stack_image_random, epoch)
            logger.add_image('image_fix', stack_image_fix, epoch)
        torch.save(encoder.state_dict(), '{}/models/encoder-epoch{:0>3d}.pth'.format(dir_save, epoch))
        torch.save(decoder.state_dict(), '{}/models/decoder-epoch{:0>3d}.pth'.format(dir_save, epoch))
        if args.use_noise:
            torch.save(noiser.state_dict(), '{}/models/noiser-epoch{:0>3d}.pth'.format(dir_save, epoch))
        torch.save(discriminator.state_dict(), '{}/models/discriminator-epoch{:0>3d}.pth'.format(dir_save, epoch))


if __name__ == '__main__':
    args = parse()
    train(args)
