

from copy import deepcopy
import types
import torch
from torch.optim import SGD
from utils.afd import MultiTaskAFDAlternative
from utils.augmentations import CustomRandomCrop, CustomRandomHorizontalFlip, DoubleCompose, DoubleTransform
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

from torchvision import transforms
import torch.nn.functional as F


def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Double-branch distillation + inter-branch skip attention')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)

    # Griddable parameters
    parser.add_argument('--der_alpha', type=float, required=False,default=0.3,
                        help='Distillation alpha hyperparameter for student stream.')
    parser.add_argument('--der_beta', type=float, required=False,default=0.9,
                        help='Distillation beta hyperparameter.')
    parser.add_argument('--lambda_fp', type=float, required=False,default=0.1,
                        help='weight of feature propagation loss replay') # 特征传播损失重放权重

    parser.add_argument('--lambda_diverse_loss', type=float, required=False, default=0,
                        help='Diverse loss hyperparameter.')
    parser.add_argument('--lambda_fp_replay', type=float, required=False, default=0,
                        help='weight of feature propagation loss replay')
    parser.add_argument('--resize_maps', type=int, required=False, choices=[0, 1], default=0,
                        help='Apply downscale and upscale to feature maps before save in buffer?') 
    parser.add_argument('--min_resize_threshold', type=int, required=False, default=16,
                        help='Min size of feature maps to be resized?')

    return parser


class TwF(ContinualModel):

    NAME = 'twf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(
            backbone, loss, args, transform) # 先继承自ContinualModel的初始化

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.buf_transform = self.get_custom_double_transform(self.transform.transforms)

        ds = get_dataset(args) # 主要为了获取参数，不参与训练
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_tasks = ds.N_TASKS
        self.not_aug_transform = transforms.Compose([transforms.ToPILImage(), ds.TEST_TRANSFORM]) if hasattr(ds, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), ds.get_normalization_transform()]) # hasattr(ds, 'TEST_TRANSFORM')=FALSE
        self.num_classes = self.n_tasks * self.cpt

        self.task = 0

    def get_custom_double_transform(self, transform):
        tfs = []
        for tf in transform:
            if isinstance(tf, transforms.RandomCrop):
                tfs.append(CustomRandomCrop(tf.size, tf.padding, resize=self.args.resize_maps==1, min_resize_index=2))
            elif isinstance(tf, transforms.RandomHorizontalFlip):
                tfs.append(CustomRandomHorizontalFlip(tf.p))
            elif isinstance(tf, transforms.Compose):
                tfs.append(DoubleCompose(
                    self.get_custom_double_transform(tf.transforms)))
            else:
                tfs.append(DoubleTransform(tf))

        return DoubleCompose(tfs)

    def end_task(self, dataset):
        self.eval()

        with torch.no_grad():
            # loop over buffer
            for buf_idxs in batch_iterate(len(self.buffer), self.args.batch_size):

                buf_idxs = buf_idxs.to(self.device)
                buf_labels = self.buffer.labels[buf_idxs].to(self.device)

                buf_mask = torch.div(buf_labels, self.cpt,
                                     rounding_mode='floor') == self.task

                if not buf_mask.any():
                    continue

                buf_inputs = self.buffer.examples[buf_idxs][buf_mask]
                buf_labels = buf_labels[buf_mask]
                buf_inputs = torch.stack([self.not_aug_transform(
                    ee.cpu()) for ee in buf_inputs]).to(self.device)

                _, buf_partial_features = self.net(
                    buf_inputs, returnt='full')
                prenet_input = buf_inputs
                _, pret_buf_partial_features = self.prenet(prenet_input)

                buf_partial_features = buf_partial_features[:-1]
                pret_buf_partial_features = pret_buf_partial_features[:-1]

                _, attention_masks = self.partial_distill_loss(buf_partial_features[-len(
                    pret_buf_partial_features):], pret_buf_partial_features, buf_labels)

                for idx in buf_idxs:
                    self.buffer.attention_maps[idx] = [
                        at[idx % len(at)] for at in attention_masks]

        self.train()
        self.task += 1

    def begin_task(self, dataset):

        if  self.task == 0 or ("start_from" in self.args and self.args.start_from is not None and self.task == self.args.start_from):
            self.load_aux_dataset() # 加载预训练数据集 self.aux_dset, self.aux_test_dset
            self.load_initial_checkpoint() # 先删除了上面的预训练数据，再加载预训练模型 在self.net
            # self.load_initial_checkpoint()
            # self.reset_classifier()

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

            self.prenet.forward = types.MethodType( # 定义prenet的前传
                _pret_forward, self.prenet)

            self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.num_classes).to(self.device)

            # Retrieve features
            with torch.no_grad():
                x = next(self.aux_iter)[0].to(self.device)
                _, feats_t = self.net(x, returnt='full')
                prenet_input = x
                _, pret_feats_t = self.prenet(prenet_input)

            feats_t = feats_t[:-1]
            pret_feats_t = pret_feats_t[:-1]

            for i, (x, pret_x) in enumerate(zip(feats_t, pret_feats_t)):
                # clear_grad=self.args.detach_skip_grad == 1
                adapt_shape = x.shape[1:]
                pret_shape = pret_x.shape[1:]
                if len(adapt_shape) == 1:
                    adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
                    pret_shape = (pret_shape[0], 1, 1)

                setattr(self.net, f"adapter_{i+1}", MultiTaskAFDAlternative(
                    adapt_shape, self.n_tasks, self.cpt, clear_grad=False,
                    teacher_forcing_or=False,
                    lambda_forcing_loss=self.args.lambda_fp_replay,
                    use_overhaul_fd=True, use_hard_softmax=True,
                    lambda_diverse_loss=self.args.lambda_diverse_loss, 
                    attn_mode="chsp",
                    min_resize_threshold=self.args.min_resize_threshold,
                    resize_maps=self.args.resize_maps == 1,
                ).to(self.device))

            self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                           weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            self.net.train()

            for p in self.prenet.parameters():
                p.requires_grad = False


    def partial_distill_loss(self, net_partial_features: list, pret_partial_features: list,
                             targets, teacher_forcing: list = None, extern_attention_maps: list = None):

        assert len(net_partial_features) == len(
            pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

        if teacher_forcing is None or extern_attention_maps is None:
            assert teacher_forcing is None
            assert extern_attention_maps is None

        loss = 0
        attention_maps = []

        for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
            assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}" # 判断尺寸是否一致

            adapter = getattr(
                self.net, f"adapter_{i+1}")

            pret_feat = pret_feat.detach()

            if teacher_forcing is None:
                curr_teacher_forcing = torch.zeros(
                    len(net_feat,)).bool().to(self.device)
                curr_ext_attention_map = torch.ones(
                    (len(net_feat), adapter.c)).to(self.device) # adapter.c 该层特征的通道数
            else:
                curr_teacher_forcing = teacher_forcing
                curr_ext_attention_map = torch.stack(
                    [b[i] for b in extern_attention_maps], dim=0).float()

            adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                                  teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

            loss += adapt_loss
            attention_maps.append(adapt_attention.detach().cpu().clone().data)

        return loss / (i + 1), attention_maps # 对应论文中的attn maps


    def observe(self, inputs, labels, not_aug_inputs, epoch=None): # 每个minibatch都要遍历
        labels = labels.long()
        if not hasattr(self, 'seen_so_far'):
            self.seen_so_far = torch.tensor([]).long().to(self.device) # 只是分配了一个属性：给seen_so_far赋空
        present = labels.unique() # 当前的类别，去除其中重复的元素，并按元素由小到大返回列表
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique() # 已经遇到的类别

        B = len(inputs) # batch_size大小
        all_labels = labels

        if len(self.buffer) > 0: # 第一个任务第一epoch的第一个batch还没有buffer
            # sample from buffer:buffer中的哪些，输入，标签，预测（当前模型之前的预测输出，预训练模型的之前预测输出）
            buf_choices, buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=None, return_index=True)
            buf_attention_maps = [self.buffer.attention_maps[c]
                                  for c in buf_choices]
            d = [self.buf_transform(ee.cpu(), attn_map) for ee, attn_map in zip(
                buf_inputs, buf_attention_maps)]
            buf_inputs, buf_attention_maps = torch.stack(
                [v[0] for v in d]).to(self.device), [[o.to(self.device) for o in v[1]] for v in d]
            
            inputs = torch.cat([inputs, buf_inputs]) # 长度为batch_size+minibatch_size
            all_labels = torch.cat([labels, buf_labels])
        # 返回batch_size*num_class,returnt决定的特征层（returnt='full'返回所有的,长度为6的列表）
        all_logits, all_partial_features = self.net(inputs, returnt='full') 
        prenet_input =  inputs
        all_pret_logits, all_pret_partial_features = self.prenet(prenet_input) # 返回预训练模型的相关参数,在begin_task初始化复制的
        # 预训练模型为CIFAR100，all_pret_logits维度为batch_size*100,长度为6的特征层的列表
        all_partial_features = all_partial_features[:-1] # 排除了最后一层
        all_pret_partial_features = all_pret_partial_features[:-1]

        stream_logits, buf_outputs = all_logits[:B], all_logits[B:] # 有buffer时，buf_outputs才有结果
        stream_partial_features = [p[:B] for p in all_partial_features]
        stream_pret_logits, pret_buf_logits = all_pret_logits[:B], all_pret_logits[B:]
        stream_pret_partial_features = [p[:B]
                                        for p in all_pret_partial_features]

        mask = torch.zeros_like(stream_logits)
        mask[:, present] = 1 # 当前任务的类别可见

        self.opt.zero_grad()
        # self.cpt = ds.N_CLASSES_PER_TASK
        loss = self.loss( # 交叉熵损失
            stream_logits[:, self.task*self.cpt:(self.task+1)*self.cpt], labels % self.cpt) # 选取了当前类别的列，labels映射为0/1
            # 每一行的，当前的任务的预测
        loss_er = torch.tensor(0.)
        loss_der = torch.tensor(0.)
        loss_afd = torch.tensor(0.)

        if len(self.buffer) == 0:
            loss_afd, stream_attention_maps = self.partial_distill_loss( # 返回Lfp, Attn Maps（存起来）
                stream_partial_features[-len(stream_pret_partial_features):], stream_pret_partial_features, labels)
        else:
            buffer_teacher_forcing = torch.div( # 对应元素相除
                buf_labels, self.cpt, rounding_mode='floor') != self.task # 'floor'正数向下舍去
            teacher_forcing = torch.cat(
                (torch.zeros((B)).bool().to(self.device), buffer_teacher_forcing))
            attention_maps = [
                [torch.ones_like(map) for map in buf_attention_maps[0]]]*B + buf_attention_maps

            loss_afd, all_attention_maps = self.partial_distill_loss(all_partial_features[-len(
                all_pret_partial_features):], all_pret_partial_features, all_labels,
                teacher_forcing, attention_maps)  # Lfp

            stream_attention_maps = [ap[:B] for ap in all_attention_maps]

            loss_er = self.loss(
                buf_outputs[:, :(self.task+1)*self.cpt], buf_labels) # 截取了buf_outputs的前两列

            loss_der = F.mse_loss(
                buf_outputs, buf_logits[:, :self.num_classes])

        loss += self.args.der_beta * loss_er
        loss += self.args.der_alpha * loss_der
        loss += self.args.lambda_fp * loss_afd

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=torch.cat( #batch_size*(当前任务的长度+预训练任务的长度)
                                 [stream_logits, stream_pret_logits], dim=1).data,
                             attention_maps=stream_attention_maps)
        

        return loss.item(), 0, 0, 0, 0
