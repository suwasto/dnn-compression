"""
contains function to load images to dataloader
"""

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def loader(path, input_size=224, batch_size=32, num_workers=4, pin_memory=True):
    """
    loader for the training set
    :param path: (str) path to the images folder
    :param input_size: (int) input size of the model
    :param batch_size: (int) number of batch size
    :param num_workers: (int) number of workers
    :param pin_memory: (bool) whether to pin memory
    :return:
        DataLoader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def test_loader(path, input_size=224, batch_size=32, num_workers=4, pin_memory=True):
    """
    test data loader
    :param path: (str) path to the images folder
    :param input_size: (int) input size of the model
    :param batch_size: (int) number of batch size
    :param num_workers: (int) number of workers
    :param pin_memory: (bool) whether to pin memory
    :return:
        DataLoader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
