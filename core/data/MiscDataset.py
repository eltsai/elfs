import os
import numpy as np
import torch
from pathlib import Path

from torchvision import datasets, transforms
import torchvision

from core.utils.misc import calculate_hungarian_misclassification_rate



class CIFARDataset(object):
    @staticmethod
    def load_custom_labels(dataset, label_path):
        if label_path.endswith('.pt'):
            labels = torch.load(label_path)
            assert len(labels) == len(dataset)
            # calculate misclassification rate using hungarian algorithm
            mis_cls = calculate_hungarian_misclassification_rate(labels, dataset.targets)
            print(f"Misclassification rate: {mis_cls:.4f}")
            dataset.targets = labels
            return dataset

    @staticmethod
    def get_cifar10_transform(name):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.CIFAR10
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else: raise f"Unknown augmentation method: {name}!"

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_cifar10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar10_test(path):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        return testset

    @staticmethod
    def get_cifar100_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar100_test(path):
        mean=[0.507, 0.487, 0.441]
        std=[0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset

class SVHNDataset(object):
    @staticmethod
    def load_custom_labels(dataset, label_path):
        if label_path.endswith('.pt'):
            labels = torch.load(label_path)
            assert len(labels) == len(dataset)

            mis_cls = calculate_hungarian_misclassification_rate(labels, dataset.labels)
            print(f"Misclassification rate: {mis_cls:.4f}")

            dataset.labels = labels
            return dataset
    @staticmethod
    def get_svhn_train(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform)
        return trainset

    @staticmethod
    def get_svhn_test(path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
        return testset

class CINIC10Dataset(object):
    @staticmethod
    def load_custom_labels(dataset, label_path, is_val=False, is_test=False):
        if label_path.endswith('.pt'):
            labels = torch.load(label_path)
            if is_test:
                assert len(labels) == len(dataset), "Label count does not match dataset size."
            elif is_val:
                # take last len(dataset) from labels
                labels = labels[-len(dataset):]
                print(f"Loaded {len(labels)} labels for validation set.")
            else:
                labels = labels[:len(dataset)]
                print(f"Loaded {len(labels)} labels for training set.")

            # Convert dataset labels to a tensor for comparison
            dataset_labels = torch.tensor([label for _, label in dataset.samples])
            mis_cls = calculate_hungarian_misclassification_rate(labels, dataset_labels)

            print(f"Misclassification rate: {mis_cls:.4f}")

            dataset.samples = [(path, labels[i]) for i, (path, _) in enumerate(dataset.samples)]
            return dataset
    @staticmethod
    def get_cinic10_train(path, transform=None, identity_transform=False, is_val=False):
        if transform is None:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if is_val:
            path = os.path.join(path, 'valid')
        else:
            path = os.path.join(path, 'train')
        trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        return trainset

    @staticmethod
    def get_cinic10_test(path):
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        path = os.path.join(path, 'test')
        testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        return testset

class STL10Dataset(object):
    @staticmethod
    def load_custom_labels(dataset, label_path):
        if label_path.endswith('.pt'):
            labels = torch.load(label_path)
            assert len(labels) == len(dataset)
            # calculate misclassification rate using Hungarian algorithm
            mis_cls = calculate_hungarian_misclassification_rate(labels, dataset.labels)
            print(f"Misclassification rate: {mis_cls:.4f}")
            dataset.labels = labels
            return dataset

    @staticmethod
    def get_stl10_transform(name):
        mean = [0.43, 0.42, 0.39]
        std = [0.27, 0.26, 0.27]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.IMAGENET
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else:
            raise ValueError(f"Unknown augmentation method: {name}!")

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_stl10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.43, 0.42, 0.39]
            std = [0.27, 0.26, 0.27]
            transform = transforms.Compose([
                transforms.RandomCrop(96, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.43, 0.42, 0.39]
            std = [0.27, 0.26, 0.27]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = datasets.STL10(root=path, split='train', download=False, transform=transform)
        return trainset

    @staticmethod
    def get_stl10_test(path):
        mean = [0.43, 0.42, 0.39]
        std = [0.27, 0.26, 0.27]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = datasets.STL10(root=path, split='test', download=False, transform=transform_test)
        return testset

class ImageNetDataset(object):
    @staticmethod
    def get_ImageNet_train(path, transform=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ]))


        return trainset

    @staticmethod
    def get_ImageNet_test(path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        testset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ]))
        return testset
    

class CustomImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, path, pseudo_labels=None, is_test=False):
        self.dataset = ImageNetDataset.get_ImageNet_test(path) if is_test else ImageNetDataset.get_ImageNet_train(path)
        original_labels = [x[1] for x in self.dataset.samples]
        if pseudo_labels is not None:
            # report hungarian misclassification rate
            mis_cls = calculate_hungarian_misclassification_rate(pseudo_labels, original_labels)
            print(f"Misclassification rate: {mis_cls:.4f}")
            self.pseudo_labels = pseudo_labels
        else:
            # keep the original labels
            self.pseudo_labels = original_labels

        if len(self.dataset) != len(self.pseudo_labels):
            raise ValueError(f"The dataset has {len(self.dataset)} entries but there are {len(self.pseudo_labels)} pseudo labels.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx] 
        pseudo_label = self.pseudo_labels[idx]
        return image, int(pseudo_label)
    

def get_imagenet_dataset(datapath, split='train', transform=None):
    """
    Loads the ImageNet-1K dataset from a specified path.

    Parameters:
    - datapath: Path to the ImageNet dataset.
    - split: Which dataset split to load ('train' or 'val').
    - transform: A torchvision.transforms object defining the dataset transformations.

    Returns:
    - A torchvision.datasets.ImageFolder instance for the specified dataset split.
    """

    datapath = Path(datapath)
    

    if split not in ['train', 'val']:
        raise ValueError(f"Split '{split}' is not recognized. Use 'train' or 'val'.")
    
    split_path = datapath / split
    
    dataset = datasets.ImageFolder(root=str(split_path), transform=transform)
    
    return dataset

