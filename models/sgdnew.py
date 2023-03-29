from utils.args import *
from models.utils.continual_model import ContinualModel
import types
from copy import deepcopy
from torch.nn import functional as F
import torch
from utils.cka import cka_loss
from torch.optim import SGD
from datasets import get_dataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_aux_dataset_args(parser)
    parser.add_argument('--cka', type=float, required=True,
                        help='Penalty weight.')     
    return parser


class Sgdnew(ContinualModel):
    NAME = 'sgdnew'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgdnew, self).__init__(backbone, loss, args, transform)
        self.current_task = 0

        ds = get_dataset(args) # 主要为了获取参数，不参与训练
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_tasks = ds.N_TASKS
        self.soft = torch.nn.Softmax(dim=1)
        self.num_classes = self.n_tasks * self.cpt


    def end_task(self, dataset):
        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task == 0: 
            if self.args.pre_dataset is None:
                return 0
            self.load_initial_checkpoint()
            self.reset_classifier()
            
            self.prenet = deepcopy(self.net.eval()) # 复制一个预训练的
            
            self.net.set_return_prerelu(True)
            self.prenet.set_return_prerelu(True)

            def _pret_forward(self, x):
                ret = []
                x = x.to(self.device)
                x = self.bn1(self.conv1(x))
                
                ret.append(x.clone().detach())
                x = F.relu(x)
                if hasattr(self, 'maxpool'):
                    x = self.maxpool(x)
                x = self.layer1(x)
                ret.append(self.layer1[-1].prerelu.clone().detach())
                x = self.layer2(x)
                ret.append(self.layer2[-1].prerelu.clone().detach())
                x = self.layer3(x)
                ret.append(self.layer3[-1].prerelu.clone().detach())
            
                x = self.layer4(x)
                ret.append(self.layer4[-1].prerelu.clone().detach())
                x = F.avg_pool2d(x, x.shape[2])
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                ret.append(x.clone().detach())
            
                return x, ret

            self.prenet.forward = types.MethodType(
                _pret_forward, self.prenet)

            self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.num_classes).to(self.device)

            self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                           weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            self.net.train()

            for p in self.prenet.parameters():
                p.requires_grad = False

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        labels = labels.long()
        self.opt.zero_grad()

        outputs, features = self.net(inputs, returnt='full')     # _为all_pret_logits
        loss = self.loss(outputs, labels)
        
        _, pre_features = self.prenet(inputs) # 返回预训练模型的相关参数,在begin_task初始化复制的
        features = features[:-1] # 排除了最后一层
        pre_features = pre_features[:-1]
        features = [p for p in features]
        pre_features = [p for p in pre_features]
        loss_cka = cka_loss(features[-len(pre_features):], pre_features,self.device) # 0.0158 0.02
        loss += self.args.cka*loss_cka
        loss.backward()
        self.opt.step()

        return loss.item(),0,0,0,0

