# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.misc as misc

from utils import system


def load(filename, varlist=None):
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    if varlist is None:
        outputs = data
    elif isinstance(varlist, str):
        outputs = data[varlist]
    else:
        outputs = []
        for v in varlist:
            outputs.append(data[v])
    return outputs


def save(filename, vardict):
    system.ensure_dir(filename)
    scipy.io.savemat(filename, vardict)


def show(block=False):
    plt.show(block=block)


def figure():
    h = plt.figure()
    show()
    return h


def imread(filename):
    return plt.imread(filename)


def imwrite(filename, image):
    system.ensure_dir(filename)
    plt.imsave(filename, image)


def imagesc(image, cmap='gray'):
    cmin = np.min(image[:])
    cmax = np.max(image[:])
    scaled = (image - cmin) / (cmax - cmin)
    return imshow(scaled, cmap=cmap)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def imshow(x, interpolation='nearest', cmap=None):
    if cmap is not None:
        h = plt.imshow(x, interpolation=interpolation, cmap=cmap)
    else:
        if x.ndim == 2 or x.shape[-1] == 1:
            h = plt.imshow(x, interpolation=interpolation, cmap='gray')
        else:
            h = plt.imshow(x, interpolation=interpolation)
    show()
    return h


def imresize(arr, size, interpolation='bilinear'):
    return misc.imresize(arr, size, interp=interpolation)


def subplot(shape, row, col=None,
            axis='off'):
    if col is not None:
        ind = (row - 1) * shape[1] + col
        h = plt.subplot(shape[0], shape[1], ind)
    else:
        ind = row
        row = (ind - 1) % shape[0] + 1
        col = (ind - 1) / shape[0] + 1
        ind = (row - 1) * shape[1] + col
        h = plt.subplot(shape[0], shape[1], ind)

    plt.axis(axis)
    plt.subplots_adjust(wspace=0.02, hspace=0.02,
                        bottom=0.02, top=0.98,
                        left=0.02, right=0.98)
    show()
    return h

def randn(shape):
    return np.random.standard_normal(shape).astype(np.float32)


def rand(shape):
    return np.random.random_sample(shape).astype(np.float32)
