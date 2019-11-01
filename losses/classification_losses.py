# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import torch
import torch.nn as nn

from losses import factory


class ClassificationLoss(nn.Module):
    def __init__(self, args, topk=(1, 2, 3), reduction='mean'):
        super().__init__()
        self.args = args
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.topk = topk

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, output_dict, target_dict):
        output = output_dict["output1"]
        target = target_dict["target1"]
        # compute actual losses
        cross_entropy = self.cross_entropy(output, target)
        # create dictonary for losses
        loss_dict = {
            "xe": cross_entropy,
        }
        acc_k = ClassificationLoss.accuracy(output, target, topk=self.topk)
        for acc, k in zip(acc_k, self.topk):
            loss_dict["top%i" % k] = acc
        return loss_dict


factory.register("ClassificationLoss", ClassificationLoss)
