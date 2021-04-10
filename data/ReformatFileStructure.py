import os
from PIL import Image
from torchvision import transforms
import random


def build_celeba(src='../datasets/img_align_celeba/',
                 dst='../datasets/celeba_resize/',
                 image_size=64):
    os.makedirs(dst, exist_ok=True)
    os.makedirs(dst + 'train/images', exist_ok=True)
    os.makedirs(dst + 'test/images', exist_ok=True)

    image_list = os.listdir(src)

    trans = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])

    choice_list = [i for i in range(len(image_list))]
    random.shuffle(choice_list)
    for i in range(len(image_list)):
        img = Image.open(src + image_list[choice_list[i]])
        img = trans(img)
        if i < 200000:
            img.save(dst + 'train/images/' + image_list[choice_list[i]])
        else:
            img.save(dst + 'test/images/' + image_list[choice_list[i]])
        # print(dst + image_list[choice_list[i]])


if __name__ == '__main__':
    build_celeba()
