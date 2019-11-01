# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import torch
import torch.nn.functional as tf
from torch import nn


class Meshgrid(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.width = -1
        self.height = -1
        self.xx = None
        self.yy = None

    def update_grid(self, width, height):
        if self.width != width or self.height != height:
            rangex = torch.arange(0, width)
            rangey = torch.arange(0, height)
            xx = rangex.repeat(height, 1).contiguous()
            yy = rangey.repeat(width, 1).t().contiguous()
            if self.normalize:
                xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
                yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
            self.xx = xx.view(1, 1, height, width)
            self.yy = yy.view(1, 1, height, width)
            self.width = width
            self.height = height

        return self.xx, self.yy

    def forward(self, width, height, device=None, dtype=None):
        self.xx, self.yy = self.update_grid(width=width, height=height)
        self.xx = self.xx.to(device=device, dtype=dtype)
        self.yy = self.yy.to(device=device, dtype=dtype)
        return self.xx, self.yy


class FlowWarpingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.meshgrid = Meshgrid()

    def forward(self, value, flow):
        b, c, h, w = value.size()

        xx, yy = self.meshgrid(width=w, height=h, device=value.device, dtype=value.dtype)
        grid = torch.cat((xx, yy), dim=1)

        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)

        value_warped = tf.grid_sample(value, vgrid)

        mask = value.new_ones(b, 1, h, w)
        mask_warped = (tf.grid_sample(mask, vgrid) >= 1.0).float()

        return value_warped * mask_warped


def resize2d(inputs, size_targets, mode="bilinear", align_corners=True):
    size_inputs = (inputs.size(2), inputs.size(3))

    if all([size_inputs == size_targets]):
        return inputs  # nothing to do
    elif any([size_targets < size_inputs]):
        resized = tf.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
    else:
        resized = tf.interpolate(inputs, size=size_targets, mode=mode, align_corners=align_corners)  # upsampling

    # correct scaling
    return resized


def resize2d_as(inputs, output_as, mode="bilinear", align_corners=True):
    size_targets = (output_as.size(2), output_as.size(3))
    return resize2d(inputs, size_targets, mode=mode, align_corners=align_corners)


class Interp2(nn.Module):
    def __init__(self, mode='bilinear', padding_mode='border'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, v, xq, yq):
        b, c, h, w = v.size()
        grid = torch.cat((xq, yq), dim=1).permute(0, 2, 3, 1)
        grid[:, :, :, 0] /= max(w - 1, 1)
        grid[:, :, :, 1] /= max(h - 1, 1)
        grid *= 2.0
        grid -= 1.0
        output = tf.grid_sample(v, grid, mode=self.mode, padding_mode=self.padding_mode)
        return output
