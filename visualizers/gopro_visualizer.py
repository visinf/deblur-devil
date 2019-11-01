# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import functools
import random

import torch
from torch.nn import functional as tf

from contrib.cmap import ColorMap
from contrib.middlebury import FlowToColor
from utils import summary
from visualizers import factory
from visualizers.visualizer import Visualizer


def random_indices(k, n):
    indices = list(range(n))
    random.shuffle(indices)
    return sorted(indices[0:k])


class GoProVisualizer(Visualizer):
    def __init__(self,
                 args,
                 model_and_loss,
                 optimizer,
                 param_scheduler,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 images_every_n_epochs=5,
                 keep_num_training_steps=30,
                 keep_num_validation_steps=30,
                 track_epochs=False):
        super().__init__()
        self.args = args
        self.global_train_step = 0
        self.optimizer = optimizer
        self.param_scheduler = param_scheduler
        self.lr_scheduler = lr_scheduler
        self.validation_loader = validation_loader
        self.training_loader = train_loader
        self.images_every_n_epochs = images_every_n_epochs
        self.keep_num_training_steps = keep_num_training_steps
        self.keep_num_validation_steps = keep_num_validation_steps
        self.flow2rgb = FlowToColor()
        self.err2rgb = ColorMap(cmap='jet')
        self.do_track_epochs = track_epochs
        self.model = model_and_loss.model
        self.epoch = None
        self.total_epochs = None
        self.prefix = None
        self.track_epoch = None
        self.random_step_indices = None

    def on_epoch_init(self, lr, train, epoch, total_epochs):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.prefix = 'train' if train else 'valid'
        self.random_step_indices = []

        if train:
            summary.scalar('train/lr', lr[0], global_step=epoch)

        # we track the first 2 epochs and then some others
        # self.track_epoch = False
        self.track_epoch = self.do_track_epochs and (epoch <= 2) or (epoch % self.images_every_n_epochs) == 0
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
        if train:
            self.global_train_step += 1
            for key, value in loss_dict.items():
                summary.scalar('train/%s' % key, value, global_step=self.global_train_step)

        def from_basename(name, in_basename):
            return "{}_{}/{}".format(in_basename, self.prefix, name)

        # ------------------------------------------------------------------------------
        # We also track some epochs completely on a subset of simple
        # ------------------------------------------------------------------------------
        if self.track_epoch:
            global_step = self.global_train_step if train else self.epoch
            if step in self.random_step_indices:
                batch_idx = 0
                basename = example_dict['basename'][batch_idx]
                basename = basename.replace('/', '_')
                make_name = functools.partial(from_basename, in_basename=basename)

                # visualize inputs
                input1 = example_dict['input1'][batch_idx:batch_idx + 1, ...]
                input2 = example_dict['input2'][batch_idx:batch_idx + 1, ...]
                input3 = example_dict['input3'][batch_idx:batch_idx + 1, ...]
                input4 = example_dict['input4'][batch_idx:batch_idx + 1, ...]
                input5 = example_dict['input5'][batch_idx:batch_idx + 1, ...]
                summary.images(make_name('input1'), input1, global_step=global_step)
                summary.images(make_name('input2'), input2, global_step=global_step)
                summary.images(make_name('input3'), input3, global_step=global_step)
                summary.images(make_name('input4'), input4, global_step=global_step)
                summary.images(make_name('input5'), input5, global_step=global_step)

                # visualize target
                target1 = example_dict['target1'][batch_idx:batch_idx + 1, ...]
                summary.images(make_name('gt'), target1, global_step=global_step)

                # visualize output
                output1 = model_dict['output1'][batch_idx:batch_idx + 1, ...]
                summary.images(make_name('output'), output1, global_step=global_step)

                # visualize error
                b, _, h, w = output1.size()
                error1 = torch.sum((output1 - target1) ** 2, dim=1, keepdim=True)
                error1 = self.err2rgb(error1)
                summary.images(make_name('error'), error1, global_step=global_step)

                # visualize concatenated summary image
                x1 = torch.cat((input3, target1), dim=3)
                x2 = torch.cat((error1.float() / 255, output1), dim=3)
                x = torch.cat((x1, x2), dim=2)
                summary.images(make_name('summary'), self.downsample(x), global_step=global_step)

                # visualize warped images
                if 'warped1' in model_dict.keys():
                    warped1 = model_dict['warped1'][batch_idx:batch_idx + 1, ...]
                    warped2 = model_dict['warped2'][batch_idx:batch_idx + 1, ...]
                    warped4 = model_dict['warped4'][batch_idx:batch_idx + 1, ...]
                    warped5 = model_dict['warped5'][batch_idx:batch_idx + 1, ...]
                    summary.images(make_name('warped1'), warped1, global_step=global_step)
                    summary.images(make_name('warped2'), warped2, global_step=global_step)
                    summary.images(make_name('warped4'), warped4, global_step=global_step)
                    summary.images(make_name('warped5'), warped5, global_step=global_step)

    def on_epoch_finished(self, avg_loss_dict, train, epoch, total_epochs):
        if not train:
            for key, value in avg_loss_dict.items():
                summary.scalar('valid/%s' % key, value, global_step=epoch)


factory.register('GoProVisualizer', GoProVisualizer)
