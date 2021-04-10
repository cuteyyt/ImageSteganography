import torch.utils.data
import torchvision.datasets
from torchvision import datasets, transforms


def get_data_loaders(args):
    """
    Get torch data loaders for training and validation.
    The data loaders
        take a crop of the image,
        transform it into tensor,
        and normalize it.
    """
    image_height, image_width = args.input_size
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((image_height, image_width), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = torchvision.datasets.ImageFolder(args.trainset_dir, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8)

    validation_images = torchvision.datasets.ImageFolder(args.valset_dir, transform=data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=args.validate_batch,
                                                    shuffle=False, num_workers=8)

    return train_loader, validation_loader
