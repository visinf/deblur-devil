# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import numpy as np
import torch
from torch import nn
from torch.nn import functional as tf


# The weights in this module were tested against Matlab's rgb2ycbcr function

class YCbCr(nn.Module):
    def __init__(self, clip_image=False):
        super().__init__()
        self.weight = torch.from_numpy(YCbCr.create_weights())
        self.inverse_weight = torch.from_numpy(YCbCr.create_inverse_weights())
        self.bias = torch.from_numpy(YCbCr.create_bias())
        self.clip_image = clip_image

    @staticmethod
    def create_weights():
        weight = np.array(
            [[0.256788235294118, 0.504129411764706, 0.097905882352941],
             [-0.148223529411765, -0.290992156862745, 0.439215686274510],
             [0.439215686274510, -0.367788235294118, -0.071427450980392]], np.float32)
        return YCbCr.create_convolutional_weights(weight)

    @staticmethod
    def create_inverse_weights():
        weight = np.array(
            [[1.164383561643836, 0.000000301124397, 1.596026887335704],
             [1.164383561643836, -0.391762539941450, -0.812968292162205],
             [1.164383561643836, 2.017232639556459, 0.000003054261745]], np.float32)
        return YCbCr.create_convolutional_weights(weight)

    @staticmethod
    def create_bias():
        bias = np.array(
            [[0.062745098039216], [0.501960784313725], [0.501960784313725]], np.float32)
        return bias.reshape((1, 3, 1, 1))

    @staticmethod
    def create_convolutional_weights(weights):
        noutputs = 3
        ninputs = 3
        filt = np.zeros((noutputs, ninputs, 1, 1), np.float32)
        for i in range(3):
            filt[i, ...] = weights[i, ...].reshape(3, 1, 1)
        return filt

    def forward(self, rgb):
        return self.from_rgb(rgb)

    def from_rgb(self, rgb):
        self.weight = self.weight.to(rgb.device)
        self.bias = self.bias.to(rgb.device)
        return tf.conv2d(rgb, self.weight) + self.bias

    def to_rgb(self, ycbcr):
        self.inverse_weight = self.inverse_weight.to(ycbcr.device)
        self.bias = self.bias.to(ycbcr.device)
        rgb = tf.conv2d(ycbcr - self.bias, self.inverse_weight)
        if self.clip_image:
            rgb.clamp_(0, 1)
        return rgb


# ------------------------------------------------------------------------------------
# Convert to grayscale:  0.299 * R + 0.587 * G + 0.114 * B
# ------------------------------------------------------------------------------------
class ToGrayscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.from_numpy(
            np.array([0.299, 0.587, 0.114], np.float32).reshape([1, 3, 1, 1]))

    def forward(self, inputs):
        channels = inputs.size(1)
        if channels == 3:
            self.weight = self.weight.to(inputs.device)
            return tf.conv2d(inputs, self.weight)
        elif channels == 1:
            return inputs
        else:
            warnings.warn("Number of input channels must be either 1 or 3!")
            return inputs
