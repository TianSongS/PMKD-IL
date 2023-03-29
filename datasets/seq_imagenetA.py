

from torchvision.datasets import ImageFolder,folder
import torchvision.transforms as transforms
from backbone.ResNet50 import resnet50
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders_imagenet
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torch.utils.data import DataLoader

class MyImageFolder(ImageFolder):
    """
    Overrides the ImageFolder dataset to change the getitem function.
    """
    def __init__(self, root, transform=None,
                 target_transform=None,loader=folder.default_loader) -> None:
        self.root = root
        self.not_aug_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        super(MyImageFolder, self).__init__(root, transform, target_transform,loader)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        path, target = self.data[index]

        # to return a PIL Image
        img = self.loader(path)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img
    
    # def default_loader(path):
    #     return Image.open(path).convert('RGB')


class SequentialIMAGENETA(ContinualDataset):

    NAME = 'seq-imageneta'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 50
    N_TASKS = 10
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)
    TRANSFORM = transforms.Compose(
            [transforms.RandomResizedCrop((224,224)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN,
                                  STD)])
    TEST_TRANSFORM = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN,
                                STD)])

    def get_data_loaders(self):
        train_transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        train_dataset = MyImageFolder(base_path() + 'imagenet/trainA',transform=train_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,test_transform, self.NAME)
        else:
            test_dataset = ImageFolder(base_path() + 'imagenet/valA',transform=test_transform)

        # test_loader = DataLoader(test_dataset,batch_size=16, shuffle=True, num_workers=0, drop_last=True)

        # for idx, data in test_loader:
        #     inputs, labels = data


        train, test = store_masked_loaders_imagenet(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialIMAGENETA.TRANSFORM])
        return transform
    
    @staticmethod
    def get_transform_my():
        transform = transforms.Compose(
            [SequentialIMAGENETA.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet50(SequentialIMAGENETA.N_CLASSES_PER_TASK
                        * SequentialIMAGENETA.N_TASKS) # 返回10分类的模型

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialIMAGENETA.MEAN,
                                         SequentialIMAGENETA.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialIMAGENETA.MEAN,
                                SequentialIMAGENETA.STD)
        return transform
