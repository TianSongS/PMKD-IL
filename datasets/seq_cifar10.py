

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD )])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose( # 训练集使用未增强的transforms
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True, # getitem返回
                                    download=False, transform=transform) # return img, target, not_aug_img
        if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                    download=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self) # 返回单个任务的数据
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform
    
    @staticmethod
    def get_transform_my():
        transform = transforms.Compose(
            [SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS) # 返回10分类的模型

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR10.MEAN,
                                         SequentialCIFAR10.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR10.MEAN,
                                SequentialCIFAR10.STD)
        return transform
