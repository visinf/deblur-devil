# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

# This is a PyTorch version of the flowToColor implementation in the Middlebury benchmark.
#
# To port the code, I looked at the Matlab implementation by Deqing Sun.
# Note that the Matlab implementation is itself a port of the original
# C++ implementation by Daniel Scharstein.
#
# For further details, take a look at the Middlebury benchmark:
# http://vision.middlebury.edu/flow/data/

import numpy as np
import torch
from torch import nn


def _max2d(x, keepdim=False):
    z = x.max(dim=3, keepdim=True)[0]
    z = z.max(dim=2, keepdim=True)[0]
    if keepdim:
        return z
    else:
        return z.squeeze(dim=3).squeeze(dim=2)


class FlowToColor(nn.Module):
    def __init__(self, byte=False):
        super().__init__()
        self.wheel = FlowToColor.make_wheel()
        self.eps = torch.ones(1, 1, 1, 1) * np.finfo(np.float32).eps
        self.pi = torch.ones(1, 1, 1, 1) * np.pi
        self.byte = byte

    @staticmethod
    def make_wheel():
        ry = 15
        yg = 6
        gc = 4
        cb = 11
        bm = 13
        mr = 6
        ncols = ry + yg + gc + cb + bm + mr
        colorwheel = np.zeros([ncols, 3])
        col = 0

        # ry
        colorwheel[0:ry, 0] = 255
        colorwheel[0:ry, 1] = np.transpose(np.floor(255 * np.arange(0, ry) / ry))
        col += ry

        # YG
        colorwheel[col:col + yg, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, yg) / yg))
        colorwheel[col:col + yg, 1] = 255
        col += yg

        # GC
        colorwheel[col:col + gc, 1] = 255
        colorwheel[col:col + gc, 2] = np.transpose(np.floor(255 * np.arange(0, gc) / gc))
        col += gc

        # CB
        colorwheel[col:col + cb, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cb) / cb))
        colorwheel[col:col + cb, 2] = 255
        col += cb

        # BM
        colorwheel[col:col + bm, 2] = 255
        colorwheel[col:col + bm, 0] = np.transpose(np.floor(255 * np.arange(0, bm) / bm))
        col += bm

        # MR
        colorwheel[col:col + mr, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, mr) / mr))
        colorwheel[col:col + mr, 0] = 255

        return torch.from_numpy(colorwheel.astype(np.float32))

    def compute_color(self, u, v):
        ncols, nrows = self.wheel.size()
        rad = torch.sqrt(u ** 2 + v ** 2)
        a = torch.atan2(-v, -u) / self.pi
        fk = 0.5 * (a + 1.0) * (ncols - 1.0) + 1
        k0 = torch.floor(fk).long()
        k1 = (k0 + 1).long()
        k1[k1 == ncols + 1] = 1
        f = fk - k0.float()
        images = []
        idx = rad <= 1.0
        for i in range(nrows):
            tmp = self.wheel[:, i]
            col0 = tmp[k0 - 1] / 255.0
            col1 = tmp[k1 - 1] / 255.0
            col = (1.0 - f) * col0 + f * col1
            col[idx] = 1.0 - rad[idx] * (1.0 - col[idx])
            col[~idx] = col[~idx] * 0.75
            images += [torch.floor(255.0 * col).byte()]

        result = torch.cat(images, dim=1)
        return result

    def forward(self, flow, max_flow=None):
        self.wheel = self.wheel.to(flow.device)
        self.eps = self.eps.to(flow.device)
        self.pi = self.pi.to(flow.device)

        u, v = flow.chunk(chunks=2, dim=1)

        if max_flow is not None:
            maxrad = max_flow
        else:
            rad = torch.sqrt(u ** 2 + v ** 2)
            maxrad = _max2d(rad, keepdim=True)

        u = u / (maxrad + self.eps)
        v = v / (maxrad + self.eps)

        flowim = self.compute_color(u, v)

        if self.byte:
            return flowim
        else:
            return flowim.float() / 255


def test():
    from matplotlib import pyplot as plt
    from utils import flow
    flow2col = FlowToColor()
    flow = flow.read_flo("./frame_0012.flo")
    flow = torch.from_numpy(flow).transpose(1, 2).transpose(0, 1).unsqueeze(dim=0)
    r = 15
    flow = flow[:, :, r:-r, r:-r]
    flowim = flow2col(flow).float() / 255.0
    flowim = flowim[0, ...].transpose(0, 1).transpose(1, 2).cpu().numpy()
    plt.figure()
    plt.imshow(flowim)
    plt.show()

# test()
