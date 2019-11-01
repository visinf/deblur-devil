# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import torch
import torch.nn as nn
import torch.nn.functional as tf

from contrib import weight_init
from models import factory


class LeNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.peter = nn.Parameter(torch.ones(1))
        weight_init.msra_(self.modules())

    def forward(self, example_dict):
        x = example_dict["input1"]
        x = tf.max_pool2d(tf.relu(self.conv1(x)), 2)
        x = tf.max_pool2d(tf.relu(self.conv2(x)), 2)
        x = x.view(-1, 1024)
        x = tf.relu(self.fc1(x))
        x = tf.dropout(x, training=self.training)
        x = self.fc2(x)
        output = tf.log_softmax(x, dim=1)
        return {"output1": output}


factory.register("LeNet", LeNet)
