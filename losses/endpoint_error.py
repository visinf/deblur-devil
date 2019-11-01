# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import logging

import torch
import torch.nn as nn
import torch.nn.functional as tf

from losses import factory


def elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1)


def downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])


class MultiScaleEPE(nn.Module):
    def __init__(self,
                 args,
                 scale_factor=1.0,
                 div_flow=0.05,
                 num_scales=5,
                 num_highres_scales=2,
                 coarsest_resolution_loss_weight=0.32):

        super().__init__()
        self.args = args
        self.div_flow = div_flow
        self.num_scales = num_scales
        self.scale_factor = scale_factor

        # ---------------------------------------------------------------------
        # start with initial scale
        # for "low-resolution" scales we apply a scale factor of 4
        # for "high-resolution" scales we apply a scale factor of 2
        #
        # e.g.   [0.005, 0.01, 0.02, 0.08, 0.32]
        # ---------------------------------------------------------------------
        self.weights = [coarsest_resolution_loss_weight]
        num_lowres_scales = num_scales - num_highres_scales
        for k in range(num_lowres_scales - 1):
            self.weights += [self.weights[-1] / 4]
        for k in range(num_highres_scales):
            self.weights += [self.weights[-1] / 2]
        self.weights.reverse()

        logging.value('MultiScaleEPE Weights: ', str(self.weights))
        assert (len(self.weights) == num_scales)  # sanity check

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]
            total_loss = 0.
            target_scaled = target * self.div_flow
            for j, (weight, output) in enumerate(zip(self.weights, outputs)):
                epe = elementwise_epe(output, downsample2d_as(target_scaled, output))
                total_loss += weight * epe.sum()
                loss_dict["epe%i" % (j + 2)] = epe.mean()
            loss_dict["total_loss"] = self.scale_factor * total_loss
        else:
            output = output_dict["flow1"]
            epe = elementwise_epe(output / self.div_flow, target)
            loss_dict["epe"] = epe.mean()

        return loss_dict


factory.register("MultiScaleEPE", MultiScaleEPE)
