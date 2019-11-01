# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import numpy as np
import torch
import torch.nn as nn

from augmentations import factory
from contrib.interpolation import Interp2
from contrib.interpolation import Meshgrid


def denormalize_coords(xx, yy, width, height):
    """ scale indices from [-1, 1] to [0, width/height] """
    xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
    yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
    return xx, yy


def normalize_coords(xx, yy, width, height):
    """ scale indices from [0, width/height] to [-1, 1] """
    xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
    yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
    return xx, yy


def apply_transform_to_params(theta0, theta_transform):
    a1, a2, a3, a4, a5, a6 = theta0.chunk(chunks=6, dim=1)
    b1, b2, b3, b4, b5, b6 = theta_transform.chunk(chunks=6, dim=1)
    #
    c1 = a1 * b1 + a4 * b2
    c2 = a2 * b1 + a5 * b2
    c3 = b3 + a3 * b1 + a6 * b2
    c4 = a1 * b4 + a4 * b5
    c5 = a2 * b4 + a5 * b5
    c6 = b6 + a3 * b4 + a6 * b5
    #
    new_theta = torch.cat([c1, c2, c3, c4, c5, c6], dim=1)
    return new_theta


class IdentityParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 0
        self.id = None
        self.identity_params = None

    @staticmethod
    def make_params(batch_size):
        o = torch.zeros(batch_size, 1, 1, 1)
        i = torch.ones(batch_size, 1, 1, 1)
        return torch.cat([i, o, o, o, i, o], dim=1)

    def forward(self, as_tensor):
        batch_size = as_tensor.size(0)
        if self.batch_size != batch_size:
            self.identity_params = IdentityParams.make_params(batch_size)
            self.batch_size = batch_size
        self.identity_params = self.identity_params.to(device=as_tensor.device, dtype=as_tensor.dtype)
        return self.identity_params


class RandomMirror(nn.Module):
    def __init__(self, vertical=True, p=0.5):
        super().__init__()
        self.batch_size = 0
        self.p = p
        self.vertical = vertical
        self.mirror_probs = None

    def random_sign_as(self, as_tensor):
        batch_size = as_tensor.size(0)
        if self.batch_size != batch_size:
            self.mirror_probs = torch.ones(batch_size, 1, 1, 1) * self.p
            self.mirror_probs = self.mirror_probs.to(device=as_tensor.device, dtype=as_tensor.dtype)
            self.batch_size = batch_size
        sign = torch.sign(2.0 * torch.bernoulli(self.mirror_probs) - 1.0)
        return sign

    def forward(self, theta):
        sign = self.random_sign_as(theta)
        i = torch.ones_like(sign)
        horizontal_mirror = torch.cat([sign, sign, sign, i, i, i], dim=1)
        theta *= horizontal_mirror

        # apply random sign to a4 a5 a6 (these are the guys responsible for y)
        if self.vertical:
            sign = self.random_sign_as(theta)
            vertical_mirror = torch.cat([i, i, i, sign, sign, sign], dim=1)
            theta *= vertical_mirror

        return theta


class RandomCrop(nn.Module):
    def __init__(self, crop):
        super().__init__()
        self.crop_size = crop

    def forward(self, im1, im2, flo):
        batch_size, _, height, width = im1.size()
        crop_height, crop_width = self.crop_size

        # check whether there is anything to do
        if any(self.crop_size < 1):
            return im1, im2, flo

        # get starting positions
        y0 = np.random.randint(0, height - crop_height)
        x0 = np.random.randint(0, width - crop_width)

        im1 = im1[:, :, y0:y0 + crop_height, x0:x0 + crop_width]
        im2 = im2[:, :, y0:y0 + crop_height, x0:x0 + crop_width]
        flo = flo[:, :, y0:y0 + crop_height, x0:x0 + crop_width]

        return im1, im2, flo


class RandomAffineFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.interp2 = Interp2()
        self.flow_interp2 = Interp2()
        self.meshgrid = Meshgrid()
        self.identity = IdentityParams()
        self.random_mirror = RandomMirror()
        self.xbounds = torch.from_numpy(np.array([-1, -1, 1, 1])).float().view(1, 1, 2, 2)
        self.ybounds = torch.from_numpy(np.array([-1, 1, -1, 1])).float().view(1, 1, 2, 2)

    def inverse_transform_coords(self, width, height, thetas, offset_x=None, offset_y=None):
        xx, yy = self.meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)

        if offset_x is not None:
            xx = xx + offset_x
        if offset_y is not None:
            yy = yy + offset_y

        a1, a2, a3, a4, a5, a6 = thetas.chunk(chunks=6, dim=1)

        xx, yy = normalize_coords(xx, yy, width=width, height=height)
        xq = a1 * xx + a2 * yy + a3
        yq = a4 * xx + a5 * yy + a6
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)

        return xq, yq

    def transform_coords(self, width, height, thetas):
        xx1, yy1 = self.meshgrid(width=width, height=height, device=thetas.device, dtype=thetas.dtype)
        xx, yy = normalize_coords(xx1, yy1, width=width, height=height)
        xq, yq = RandomAffineFlow.inverse_transform1(thetas, xx, yy, height=height, width=width)
        return xq, yq

    @staticmethod
    def inverse_transform1(thetas, x, y, height, width):
        a1, a2, a3, a4, a5, a6 = thetas.chunk(chunks=6, dim=1)

        z = a1 * a5 - a2 * a4
        b1 = a5 / z
        b2 = - a2 / z
        b4 = - a4 / z
        b5 = a1 / z
        #
        xhat = x - a3
        yhat = y - a6
        xq = b1 * xhat + b2 * yhat
        yq = b4 * xhat + b5 * yhat
        xq, yq = denormalize_coords(xq, yq, width=width, height=height)
        return xq, yq

    def find_invalid(self, width, height, thetas):
        xx = self.xbounds.to(thetas.device, thetas.dtype)
        yy = self.ybounds.to(thetas.device, thetas.dtype)
        #
        xq, yq = RandomAffineFlow.inverse_transform1(thetas, xx, yy, height=height, width=width)
        #
        invalid_mask = (xq < 0) | (yq < 0) | (xq >= width) | (yq >= height)
        invalid = invalid_mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) > 0
        return invalid

    def apply_random_transforms_to_params(self,
                                          theta0,
                                          max_translate,
                                          min_zoom, max_zoom,
                                          min_squeeze, max_squeeze,
                                          min_rotate, max_rotate,
                                          validate_size=None):
        max_translate *= 0.5
        batch_size = theta0.size(0)
        height, width = validate_size

        # collect valid params here
        thetas = torch.zeros_like(theta0)

        zoom = theta0.new_zeros(batch_size, 1, 1, 1)
        squeeze = torch.zeros_like(zoom)
        tx = torch.zeros_like(zoom)
        ty = torch.zeros_like(zoom)
        phi = torch.zeros_like(zoom)
        invalid = torch.ones_like(zoom)

        while invalid.sum() > 0:
            # random sampling
            zoom.uniform_(min_zoom, max_zoom)
            squeeze.uniform_(min_squeeze, max_squeeze)
            tx.uniform_(-max_translate, max_translate)
            ty.uniform_(-max_translate, max_translate)
            phi.uniform_(-min_rotate, max_rotate)

            # construct affine parameters
            sx = zoom * squeeze
            sy = zoom / squeeze
            sin_phi = torch.sin(phi)
            cos_phi = torch.cos(phi)
            b1 = cos_phi * sx
            b2 = sin_phi * sy
            b3 = tx
            b4 = - sin_phi * sx
            b5 = cos_phi * sy
            b6 = ty

            theta_transform = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
            theta_try = apply_transform_to_params(theta0, theta_transform)
            thetas = invalid * theta_try + (1 - invalid) * thetas

            # compute new invalid ones
            invalid = self.find_invalid(width=width, height=height, thetas=thetas).float()

        # here we should have good thetas within borders
        return thetas

    def transform_image(self, images, thetas):
        batch_size, channels, height, width = images.size()
        xq, yq = self.transform_coords(width=width, height=height, thetas=thetas)
        transformed = self.interp2(images, xq, yq)
        return transformed

    def transform_flow(self, flow, theta1, theta2):
        batch_size, channels, height, width = flow.size()
        u, v = flow.chunk(chunks=2, dim=1)

        # inverse transform coords
        x0, y0 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta1)

        x1, y1 = self.inverse_transform_coords(
            width=width, height=height, thetas=theta2, offset_x=u, offset_y=v)

        # subtract and create new flow
        u = x1 - x0
        v = y1 - y0
        new_flow = torch.cat((u, v), dim=1)

        # transform coords
        xq, yq = self.transform_coords(width=width, height=height, thetas=theta1)

        transformed = self.flow_interp2(new_flow, xq, yq)

        return transformed

    def forward(self, example_dict):
        im1 = example_dict["input1"]
        im2 = example_dict["input2"]
        flo = example_dict["target1"]

        batch_size, _, height, width = im1.size()

        # identity = no transform
        theta0 = self.identity(im1)

        # global transform
        theta1 = self.apply_random_transforms_to_params(
            theta0,
            max_translate=0.2,
            min_zoom=1.0, max_zoom=1.5,
            min_squeeze=0.86, max_squeeze=1.16,
            min_rotate=-0.2, max_rotate=0.2,
            validate_size=[height, width])

        # random flip images
        theta1 = self.random_mirror(theta1)

        # relative transform
        theta2 = self.apply_random_transforms_to_params(
            theta1,
            max_translate=0.015,
            min_zoom=0.985, max_zoom=1.015,
            min_squeeze=0.98, max_squeeze=1.02,
            min_rotate=-0.015, max_rotate=0.015,
            validate_size=[height, width])

        im1 = self.transform_image(im1, theta1)
        im2 = self.transform_image(im2, theta2)
        flo = self.transform_flow(flo, theta1, theta2)

        # construct updated dictionaries
        example_dict["input1"] = im1
        example_dict["input2"] = im2
        example_dict["target1"] = flo

        return example_dict


factory.register("RandomAffineFlow", RandomAffineFlow)
