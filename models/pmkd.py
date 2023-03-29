import types
from copy import deepcopy
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from utils.cka import cka_loss
from torch.optim import SGD
from datasets import get_dataset
import time

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--cka', type=float, required=True,
                        help='Penalty weight.')                    
    return parser


class PMKD(ContinualModel):
    NAME = 'pmkd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(PMKD, self).__init__(backbone, loss, args, transform)
        ds = get_dataset(args) 
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_tasks = ds.N_TASKS
        self.num_classes = self.n_tasks * self.cpt
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.soft = torch.nn.Softmax(dim=1)

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_aux_dataset() 
            self.load_initial_checkpoint()
            # self.reset_classifier()
            self.prenet = deepcopy(self.net.eval()) 
            
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
            
    def end_task(self, dataset):
        self.current_task += 1


    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        self.opt.zero_grad()
        # outputs = self.net(inputs).float()
        outputs, features = self.net(inputs, returnt='full')    
        loss = self.loss(outputs, labels) # 0.1646 0.1978

        _, pre_features = self.prenet(inputs) 
        features = features[:-1] 
        pre_features = pre_features[:-1]
        features = [p for p in features]
        pre_features = [p for p in pre_features]
        
        loss_cka = 0

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs).float()
            loss_alpha = F.mse_loss(buf_outputs, buf_logits) 

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs).float()
            loss_beta = self.loss(buf_outputs, buf_labels) 
            
            loss += self.args.alpha *  loss_alpha
            loss += self.args.beta *  loss_beta
        
        loss += self.args.cka * loss_cka

        loss.backward()
        self.opt.step()


        self.buffer.add_data(examples=not_aug_inputs,
                            labels=labels,
                            logits=outputs.data)

        return loss.item(),0,0,0,0
