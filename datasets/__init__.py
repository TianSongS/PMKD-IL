

from datasets.seq_cub200 import SequentialCUB200
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.seq_imagenetA import SequentialIMAGENETA
from datasets.seq_imagenetB import SequentialIMAGENETB
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    SequentialCUB200.NAME: SequentialCUB200,
    SequentialIMAGENETA.NAME: SequentialIMAGENETA,
    SequentialIMAGENETB.NAME: SequentialIMAGENETB,

}

def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys() # 检查数据集是否已定义
    return NAMES[args.dataset](args) # 为返回的对象赋值