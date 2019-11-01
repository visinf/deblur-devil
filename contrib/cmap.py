# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import numpy as np
import torch
from matplotlib import cm
from torch import nn

# ----------------------------------------------------------------------------------------
# See https://matplotlib.org/examples/color/colormaps_reference.html
#
# Typical choices are: 'gray', jet', 'viridis', 'hot'
# ----------------------------------------------------------------------------------------

COLORMAPS = [

    # Perceptually Uniform Sequential
    'viridis', 'plasma', 'inferno', 'magma',

    # Sequential
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',

    # Sequential (2)
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',

    # Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',

    # Qualitative,
    'Pastel1', 'Pastel2', 'Paired', 'Accent',
    'Dark2', 'Set1', 'Set2', 'Set3',
    'tab10', 'tab20', 'tab20b', 'tab20c',

    # Miscellaneous
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'

]


class ColorMap(nn.Module):
    #
    # Note: uint8 inputs are never normalized.
    #       float inputs are normalized if normalize_floats=True
    #
    def __init__(self, cmap='jet', normalize_floats=True, output_dtype=torch.uint8):
        super().__init__()
        if cmap not in COLORMAPS:
            raise ValueError('Unknown colormap!')
        self.normalize_floats = normalize_floats
        self.cmap = torch.from_numpy(self.get_cmap_as_float_array(cmap)).view(-1, 3)
        if output_dtype == torch.uint8:
            self.cmap = (255 * self.cmap).byte()

    @staticmethod
    def get_cmap_as_float_array(cmap_name):
        raw_cmap = cm.get_cmap(cmap_name, 256)
        cmap_array = raw_cmap(np.arange(256))[:, 0:3]  # remove alpha channels
        return cmap_array

    @staticmethod
    def min2d(tensor):
        b, c, h, w = tensor.size()
        return tensor.view(b, c, h * w).min(dim=2, keepdim=True)[0].unsqueeze(dim=3)

    @staticmethod
    def max2d(tensor):
        b, c, h, w = tensor.size()
        return tensor.view(b, c, h * w).max(dim=2, keepdim=True)[0].unsqueeze(dim=3)

    def forward(self, value):
        b, c, h, w = value.size()
        assert c == 1, 'ColorMap expects second dimension of size 1L'
        if not isinstance(value, torch.ByteTensor):
            if self.normalize_floats:
                cmin = self.min2d(value)
                cmax = self.max2d(value)
                normalized = (value - cmin) / torch.max(cmax - cmin, torch.ones_like(value) * 1e-5)
                normalized = (normalized * 255).long()
            else:
                normalized = (value * 255).long()
        else:
            normalized = value.long()
        self.cmap = self.cmap.to(value.device)
        z = torch.index_select(self.cmap, dim=0, index=normalized.view(-1))
        return z.transpose(0, 1).contiguous().view(b, 3, h, w)
