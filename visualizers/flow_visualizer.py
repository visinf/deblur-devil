# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import functools
import random

import numpy as np
import torch
from torch.nn import functional as tf

from contrib.middlebury import FlowToColor
from utils import summary
from visualizers import factory
from visualizers.visualizer import Visualizer


def random_indices(k, n):
    indices = list(range(n))
    random.shuffle(indices)
    return sorted(indices[0:k])


def downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])


def max2d(x, keepdim=False):
    z = x.max(dim=3, keepdim=True)[0]
    z = z.max(dim=2, keepdim=True)[0]
    if keepdim:
        return z
    else:
        return z.squeeze(dim=3).squeeze(dim=2)


def min2d(x, keepdim=False):
    z = x.min(dim=3, keepdim=True)[0]
    z = z.min(dim=2, keepdim=True)[0]
    if keepdim:
        return z
    else:
        return z.squeeze(dim=3).squeeze(dim=2)


def max3d(x, keepdim=False):
    z = x.max(dim=3, keepdim=True)[0]
    z = z.max(dim=2, keepdim=True)[0]
    z = z.max(dim=1, keepdim=True)[0]
    if keepdim:
        return z
    else:
        return z.squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)


def min3d(x, keepdim=False):
    z = x.min(dim=3, keepdim=True)[0]
    z = z.min(dim=2, keepdim=True)[0]
    z = z.min(dim=1, keepdim=True)[0]
    if keepdim:
        return z
    else:
        return z.squeeze(dim=3).squeeze(dim=2).squeeze(dim=1)


class FlowVisualizer(Visualizer):
    def __init__(self,
                 args,
                 model_and_loss,
                 optimizer,
                 param_scheduler,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 images_every_n_epochs=8,
                 keep_num_training_steps=30,
                 keep_num_validation_steps=30):
        super().__init__()
        self.args = args
        self.model_and_loss = model_and_loss
        self.optimizer = optimizer
        self.param_scheduler = param_scheduler
        self.lr_scheduler = lr_scheduler
        self.global_train_step = 0
        self.training_loader = train_loader
        self.validation_loader = validation_loader
        self.images_every_n_epochs = images_every_n_epochs
        self.keep_num_training_steps = keep_num_training_steps
        self.keep_num_validation_steps = keep_num_validation_steps
        self.flow2rgb = FlowToColor(byte=False)
        self.epoch = None
        self.total_epochs = None
        self.track_epoch = None
        if train_loader is not None:
            self.num_train_steps = len(train_loader)
        else:
            self.num_train_steps = 0
        if validation_loader is not None:
            self.num_valid_steps = len(validation_loader)
            self.random_step_indices = np.linspace(
                start=0, stop=self.num_valid_steps - 1,
                num=keep_num_validation_steps, dtype=np.int32)
        else:
            self.num_valid_steps = 0
            self.random_step_indices = []

    def on_epoch_init(self, lr, train, epoch, total_epochs):
        self.epoch = epoch
        self.total_epochs = total_epochs

        if train:
            summary.scalar('train/lr', lr[0], global_step=epoch)

        # we track the first 2 epochs and then some others
        self.track_epoch = (epoch <= 2) or (epoch % self.images_every_n_epochs) == 0
        if self.track_epoch:
            if train:
                if self.keep_num_training_steps > 0:
                    self.random_step_indices = random_indices(
                        k=self.keep_num_training_steps,
                        n=len(self.training_loader))
            else:
                if self.keep_num_validation_steps > 0:
                    self.random_step_indices = random_indices(
                        k=self.keep_num_validation_steps,
                        n=len(self.validation_loader))

    @staticmethod
    def downsample(x, rscale=2):
        h, w = x.size()[2:4]
        h0, w0 = [s // rscale for s in [h, w]]
        return tf.interpolate(x, (h0, w0))

    def on_step_finished(self, example_dict, model_dict, loss_dict, train, step, total_steps):

        prefix = 'train' if train else 'valid'

        def from_basename(name, in_basename):
            return "{}/{}/{}".format(prefix, in_basename, name)

        if train:
            self.global_train_step += 1
            for key, value in loss_dict.items():
                summary.scalar('train/%s' % key, value, global_step=self.global_train_step)

        if self.track_epoch:
            global_step = self.global_train_step if train else self.epoch
            if step in self.random_step_indices:
                basename = example_dict['basename'][0]
                basename = basename.replace('/', '_')
                make_name = functools.partial(from_basename, in_basename=basename)

                input1 = example_dict['input1']
                input2 = example_dict['input2']
                target1 = example_dict['target1']

                progress = [input1, input2]

                if train:
                    flow = model_dict['flow2']
                    target1 *= self.args.loss_div_flow
                    target1 = downsample2d_as(target1, flow)
                else:
                    flow = model_dict['flow1']
                    flow /= self.args.loss_div_flow

                u, v = target1.chunk(chunks=2, dim=1)
                rad = torch.sqrt(u ** 2 + v ** 2)
                max_flow = max2d(rad, keepdim=True)
                max_flow = max_flow.repeat(2, 1, 1, 1)

                flowim, targetim = self.flow2rgb(
                    torch.cat((flow, target1), dim=0), max_flow=max_flow).chunk(chunks=2, dim=0)
                size = tuple(flow.size()[2:4])

                progress = [tf.interpolate(im, size, mode='bilinear', align_corners=True) for im in progress]

                progress.append(flowim)
                progress.append(targetim)

                progress = torch.cat(progress, dim=-1)

                if progress.size(2) != 128:
                    factor = 128 / progress.size(2)
                    new_height = int(progress.size(2) * factor)
                    new_width = int(progress.size(3) * factor)
                    new_size = (new_height, new_width)
                    progress = tf.interpolate(progress, size=new_size, mode='nearest')

                progress = progress[0, ...]

                summary.image(make_name('progress'), progress, global_step=global_step)

    def on_epoch_finished(self, avg_loss_dict, train, epoch, total_epochs):
        if not train:
            for key, value in avg_loss_dict.items():
                summary.scalar('valid/%s' % key, value, global_step=epoch)


factory.register('FlowVisualizer', FlowVisualizer)
