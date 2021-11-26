from celeba_dataset import CelebADataset
from pets_dataset import PetsDataset
from transforms import get_transforms

import torch
import torchvision

def get_dataloader(dataset, split, data_dir, is_contrastive=False, n_views=2, img_size=224, batch_size=32, num_workers=4):
    if split not in ["train", "test", "valid", "val1", "val"]:
        raise ValueError("Invalid split: {}".format(split))
    if dataset == "cifar10":
        if split not in ["train", "test"]:
            raise ValueError("{} doesnt have split {}".format(dataset, split))
        data = _get_cifar10_data(split, data_dir, is_contrastive=is_contrastive, n_views=n_views, img_size=img_size)
    elif dataset == "celeba":
        data = _get_celeba_data(split, is_contrastive=is_contrastive, n_views=n_views, img_size=img_size)
    elif dataset == "oxfordpets":
        data = _get_oxford_pets_data(split, is_contrastive=is_contrastive, n_views=n_views, img_size=img_size)
    elif dataset == "imagenet":
        data = _get_imagenet_data(split, is_contrastive=is_contrastive, n_views=n_views, img_size=img_size)
    else:
        raise ValueError("Invalid dataset name: {}".format(dataset))
    dl = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dl

def _get_cifar10_data(split, data_dir, is_contrastive=False, n_views=2, img_size=32):
    if split == "train":
        isTrain = True
    else:
        isTrain = False
    transform = get_transforms(is_contrastive, img_size, n_views)
    cifar_data = torchvision.datasets.CIFAR10(root=data_dir, train=isTrain, download=True, transform=transform)
    return cifar_data

def _get_celeba_data(split, is_contrastive=False, n_views=2, img_size=224):
    dataset = CelebADataset( '/srv/share/akrishna/bias-discovery/celeba/list_attr_celeba.txt',
                            '/srv/share/akrishna/bias-discovery/celeba/img_align_celeba/',
                            transform = get_transforms(is_contrastive, img_size, n_views))
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = indices[:162770], indices[162771:182637], indices[182638:]

    if split == 'train':
        dataset = torch.utils.data.Subset(dataset, train_indices)
    elif split == 'test':
        dataset = torch.utils.data.Subset(dataset, test_indices)
    return dataset

def _get_oxford_pets_data(split, is_contrastive=False, n_views=2, img_size=224):
    path_to_data = '/srv/share/datasets/oxford_pets/images'
    transform = get_transforms(is_contrastive, img_size, n_views)

    dataset = PetsDataset(path_to_data, transform=transform)
    indices = list(range(len(dataset)))
    test_pct = 0.1
    test_size = int(test_pct * len(dataset))
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]

    if split == 'train':
        dataset = torch.utils.data.Subset(dataset, train_indices)
    elif split == 'test':
        dataset = torch.utils.data.Subset(dataset, test_indices)
    return dataset

def _get_imagenet_data(split, is_contrastive=False, n_views=2, img_size=224):
    transform = get_transforms(is_contrastive, img_size, n_views)
    if split == 'test':
        split = 'val1'
    imagenet_data = torchvision.datasets.ImageFolder(root="/srv/share/datasets/ImageNet/"+split+'/', transform=transform)
    return imagenet_data
