# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import numbers
import random
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import Lambda
from torchvision.transforms import functional as tf

from . import functional as my_tf


class PILImageToNDArray:
    def __call__(self, image):
        return my_tf.pil_to_ndarray(image)


class Identity:
    def __init__(self):
        pass

    def __call__(self, *args):
        return args


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: my_tf.adjust_brightness(img, brightness_factor)))
        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: my_tf.adjust_contrast(img, contrast_factor)))
        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: my_tf.adjust_saturation(img, saturation_factor)))
        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: my_tf.adjust_hue(img, hue_factor)))
        random.shuffle(transforms)
        transform = Compose(transforms)
        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)


class RandomMultiplicativeColor:
    def __init__(self, min_factor=0.5, max_factor=2.0, clip_image=True):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.clip_image = clip_image

    def __call__(self, image):
        return my_tf.image_random_uniform_color(
            image,
            min_factor=self.min_factor,
            max_factor=self.max_factor,
            clip_image=self.clip_image)


class RandomUniformGamma:
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    def __call__(self, image):
        return my_tf.image_random_uniform_gamma(
            image,
            min_gamma=self._min_gamma,
            max_gamma=self._max_gamma,
            clip_image=self._clip_image)


class RandomNoise:
    def __init__(self, min_stddev, max_stddev, clip_image=False):
        self._min_stddev = min_stddev
        self._max_stddev = max_stddev
        self._clip_image = clip_image

    def __call__(self, image):
        return my_tf.image_random_uniform_noise(
            image,
            min_stddev=self._min_stddev,
            max_stddev=self._max_stddev,
            clip_image=self._clip_image)


class RandomNoiseFromNormal:
    def __init__(self, stddev=2 / 255, clip_image=True):
        self.stddev = stddev
        self.clip_image = clip_image

    def __call__(self, image):
        return my_tf.image_random_noise_from_normal(
            image, stddev=self.stddev, clip_image=self.clip_image)


class RandomGaussianPhotometrics:
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, im):
        im = my_tf.image_random_gaussian_color(im, stddev=self.stddev, clip_image=True)
        im = my_tf.image_random_gaussian_gamma(im, stddev=self.stddev, clip_image=False)
        im = my_tf.image_random_gaussian_brightness(im, stddev=self.stddev, clip_image=False)
        im = my_tf.image_random_gaussian_contrast(im, stddev=self.stddev, clip_image=True)
        return im


class MultiImageRandomResizedCrop(object):
    def __init__(self, size,
                 mode='random_choice',
                 scale_choices=(1 / 4, 1 / 3, 1 / 2),
                 scale_range=(1 / 4, 1.0),
                 interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.scale_choices = scale_choices
        self.scale_range = scale_range
        self.mode = mode
        if mode not in ['random_choice', 'random_range', 'none']:
            raise ValueError("mode argument is not valid. Options are: 'random_choice' or 'random_range' or 'none'")

    def apply_random_scale(self, images):
        if self.mode == 'none':
            return images

        elif self.mode == 'random_choice':
            s = random.choice(self.scale_choices)

        else:
            s = random.uniform(self.scale_range[0], self.scale_range[1])

        im0_w, im0_h = images[0].size
        hs = int(round(im0_h * s))
        ws = int(round(im0_w * s))
        result = [x.resize((ws, hs), self.interpolation) for x in images]
        # w, h = result[0].size
        return result

    def apply_random_crop(self, images):
        h, w = self.size
        im0_w, im0_h = images[0].size
        y0 = np.random.randint(0, im0_h - h)
        x0 = np.random.randint(0, im0_w - w)
        return [x.crop((x0, y0, x0 + w, y0 + h)) for x in images]

    def __call__(self, *args):
        images = self.apply_random_scale(args)
        images = self.apply_random_crop(images)
        return images[0] if len(args) == 1 else images


class MultiImageRandomCrop(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def apply_random_crop(self, images):
        h, w = self.size
        im0_w, im0_h = images[0].size
        y0 = np.random.randint(0, im0_h - h)
        x0 = np.random.randint(0, im0_w - w)
        return [x.crop((x0, y0, x0 + w, y0 + h)) for x in images]

    def __call__(self, *args):
        images = self.apply_random_crop(args)
        return images[0] if len(args) == 1 else images


class MultiImageCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def apply_central_crop(self, images):
        h, w = self.size
        im0_w, im0_h = images[0].size

        y0 = im0_h // 2 - h // 2
        x0 = im0_w // 2 - w // 2

        return [x.crop((x0, y0, x0 + w, y0 + h)) for x in images]

    def __call__(self, *args):
        images = self.apply_central_crop(args)
        return images[0] if len(args) == 1 else images


# class MultiImageRandomResizedCrop(object):
#     def __init__(self, size,
#                  scale_choices=(1/4, 1/3, 1/2),
#                  ratio=(1.0, 1.0),
#                  interpolation=Image.BICUBIC):
#         self.size = size
#         self.interpolation = interpolation
#         self.scale_choices = scale_choices
#         self.ratio = ratio

#     @staticmethod
#     def get_params(img, scale, ratio):
#         area = img.size[0] * img.size[1]
#         for attempt in range(10):
#             target_area = random.uniform(*scale) * area
#             aspect_ratio = random.uniform(*ratio)
#             w = int(round(np.sqrt(target_area * aspect_ratio)))
#             h = int(round(np.sqrt(target_area / aspect_ratio)))
#             if random.random() < 0.5:
#                 w, h = h, w
#             if w <= img.size[0] and h <= img.size[1]:
#                 i = random.randint(0, img.size[1] - h)
#                 j = random.randint(0, img.size[0] - w)
#                 return i, j, h, w
#         # Fallback
#         w = min(img.size[0], img.size[1])
#         i = (img.size[1] - w) // 2
#         j = (img.size[0] - w) // 2
#         return i, j, w, w

#     def __call__(self, *images):
#         scale = np.random.choice(self.scale_choices)
#         i, j, h, w = self.get_params(images[0], (scale, scale), self.ratio)
#         w1, h1 = images[0].size
#         if self.size[0] > 0:
#             h1 = self.size[0]
#         if self.size[1] > 0:
#             w1 = self.size[1]
#         return [tf.resized_crop(img, i, j, h, w, (h1, w1), self.interpolation) for img in images]


class RandomChoice:
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *img):
        t = random.choice(self.transforms)
        return t(*img)


class RandomChannelPermutation(object):
    def __call__(self, image):
        return my_tf.image_random_channel_permutation(image)


class MultiImageRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            images = [tf.vflip(img) for img in args]
        else:
            images = args
        return images[0] if len(args) == 1 else images


class MultiImageRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *args):
        if random.random() < self.p:
            images = [tf.hflip(img) for img in args]
        else:
            images = args
        return images[0] if len(args) == 1 else images


class MultiImageRandomRotation(object):
    def __init__(self, degrees, resample=Image.NEAREST, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, *args):
        angle = self.get_params(self.degrees)
        result = []
        for img in args:
            im = tf.rotate(img, angle, self.resample, self.expand, self.center)
            result.append(im)
        return result[0] if len(args) == 1 else result


class RandomCropArrays(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(ndarray, output_size):
        h, w = ndarray.shape[0:2]
        th, tw = output_size
        y = random.randint(0, h - th)
        x = random.randint(0, w - tw)
        return y, x, th, tw

    @staticmethod
    def crop(ndarray, y, x, h, w):
        return ndarray[y:y + h, x:x + w, :]

    def __call__(self, *args):
        y, x, h, w = self.get_params(args[0], self.size)
        images = [self.crop(arr, y, x, h, w) for arr in args]
        return images[0] if len(args) == 1 else images


# ------------------------------------------------------------------
# Allow transformation chains of the type:
#   im1, im2, .... = transform(im1, im2, ...)
# ------------------------------------------------------------------
class TransformChainer:
    def __init__(self, list_of_transforms):
        if not isinstance(list_of_transforms, list):
            self.list_of_transforms = [list_of_transforms]
        else:
            self.list_of_transforms = list_of_transforms

    def __call__(self, *args):
        list_of_args = list(args)
        for transform in self.list_of_transforms:
            list_of_args = [transform(arg) for arg in list_of_args]
        return list_of_args[0] if len(args) == 1 else list_of_args


# ------------------------------------------------------------------
# Allow transformation chains of the type:
#   im1, im2, .... = split( transform( concatenate(im1, im2, ...) ))
# ------------------------------------------------------------------
def pil_to_ndarray8(image):
    return np.array(image, dtype=np.uint8)


class ConcatTransformSplitChainer:
    def __init__(self, list_of_transforms):
        self.chainer = TransformChainer(list_of_transforms)

    def ndarray8_to_pil(self, ndarray):
        return Image.fromarray(ndarray, 'RGB')

    def __call__(self, *args):
        num_splits = len(args)
        if num_splits == 1:
            warnings.warn("ConcatTransformSplitChainer should only be used with multiple images!")

        # --------------------------------------------------------------------------
        # Depending on the input type (numpy, torch, PIL) we concatenate inputs
        # in the first axis.
        # --------------------------------------------------------------------------
        if isinstance(args[0], np.ndarray):
            # concatenate ndarray
            concatenated = np.concatenate(args, axis=0)

        elif isinstance(args[0], torch.Tensor):
            # concatenate torch tensor
            concatenated = torch.cat(args, dim=1)
        else:
            # concatenate PIL images through numpy ndarrays
            ndarrays = [pil_to_ndarray8(img) for img in args]
            concatenated = self.ndarray8_to_pil(
                np.concatenate(ndarrays, axis=0))

        transformed = self.chainer(concatenated)

        # --------------------------------------------------------------------------
        # After transformation we split them again
        # --------------------------------------------------------------------------
        if isinstance(transformed, np.ndarray):
            # split ndarray
            split = np.split(transformed, indices_or_sections=num_splits, axis=0)
        elif isinstance(transformed, torch.Tensor):
            # split torch Tensor
            split = torch.chunk(transformed, num_splits, dim=1)
        else:
            # split PIL image through numpy ndarrays
            ndarray = pil_to_ndarray8(transformed)
            split = np.split(ndarray, indices_or_sections=num_splits, axis=0)
            split = [self.ndarray8_to_pil(x) for x in split]

        return split
