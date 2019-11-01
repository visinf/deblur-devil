# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import functools

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tf


def pil_to_ndarray(image):
    return np.array(image, dtype=np.float32) / 255.0


def ndarray_to_pil(ndarray):
    return Image.fromarray((ndarray * 255).astype('uint8'), 'RGB')


def tensor_to_ndarray(tensor):
    return np.transpose(tensor.numpy(), (1, 2, 0))


def ndarray_to_tensor(ndarray):
    return torch.from_numpy(np.transpose(ndarray, (2, 0, 1)))


def caffe_random_normal(mean=0.0, stddev=1.0, exp=False):
    result = np.random.normal(mean, stddev)
    if exp:
        result = np.exp(result)
    if isinstance(result, np.ndarray):
        result = result.astype(np.float32)
    return result


def caffe_random_uniform(low=0.0, high=1.0, size=None, exp=False):
    result = np.random.uniform(low=low, high=high, size=size)
    if exp:
        result = np.exp(result)
    if isinstance(result, np.ndarray):
        result = result.astype(np.float32)
    return result


class Functor:
    def __init__(self, torch_func=None, numpy_func=None, pil_func=None, clip_image=False):
        self.torch_func = torch_func
        self.numpy_func = numpy_func
        self.pil_func = pil_func
        self.clip_image = clip_image

    def __call__(self, image):

        if isinstance(image, torch.Tensor):
            if self.torch_func is not None:
                image = self.torch_func(image)
                if self.clip_image:
                    image.clamp_(0.0, 1.0)
                return image

            elif self.pil_func is not None:
                pil_image = ndarray_to_pil(tensor_to_ndarray(image))
                transformed = self.pil_func(pil_image)
                ndarray = pil_to_ndarray(transformed)
                return ndarray_to_tensor(ndarray)

        if isinstance(image, np.ndarray):
            if self.numpy_func is not None:
                image = self.numpy_func(image)
                if self.clip_image:
                    image = np.clip(image, 0.0, 1.0)
                return image

            elif self.pil_func is not None:
                pil_image = ndarray_to_pil(image)
                transformed = self.pil_func(pil_image)
                return pil_to_ndarray(transformed)

        if isinstance(image, Image.Image):
            if self.pil_func is not None:
                return self.pil_func(image)
            elif self.numpy_func is not None:
                image = pil_to_ndarray(image)
                transformed = self.numpy_func(image)
                if self.clip_image:
                    transformed = np.clip(transformed, 0.0, 1.0)
                pil_image = ndarray_to_pil(transformed)
                return pil_image

        raise ValueError('No handler found for transform functor. Given: {}'.format(type(image)))


def image_random_channel_permutation(image):
    def convert_torch(tensor):
        num_channels = tensor.size(0)
        indices = np.random.permutation(range(num_channels))
        return tensor[indices, :, :]

    def convert_numpy(ndarray):
        num_channels = ndarray.shape[2]
        indices = np.random.permutation(range(num_channels))
        return ndarray[:, :, indices]

    return Functor(torch_func=convert_torch,
                   numpy_func=convert_numpy)(image)


def adjust_brightness(img, brightness_factor, clip_image=True):
    def convert_tensor_or_array(tensor_or_ndarray, inner_brightness_factor):
        tensor_or_ndarray *= inner_brightness_factor
        return tensor_or_ndarray

    return Functor(torch_func=functools.partial(convert_tensor_or_array, inner_brightness_factor=brightness_factor),
                   numpy_func=functools.partial(convert_tensor_or_array, inner_brightness_factor=brightness_factor),
                   pil_func=functools.partial(tf.adjust_brightness, brightness_factor=brightness_factor),
                   clip_image=clip_image)(img)


def adjust_contrast(img, contrast_factor, clip_image=True):
    def convert_tensor_or_array(tensor_or_ndarray, inner_contrast_factor):
        return inner_contrast_factor * (tensor_or_ndarray - 0.5) + 0.5

    return Functor(torch_func=functools.partial(convert_tensor_or_array, inner_contrast_factor=contrast_factor),
                   numpy_func=functools.partial(convert_tensor_or_array, inner_contrast_factor=contrast_factor),
                   pil_func=functools.partial(tf.adjust_contrast, contrast_factor=contrast_factor),
                   clip_image=clip_image)(img)


def adjust_saturation(img, saturation_factor, clip_image=True):
    return Functor(pil_func=functools.partial(tf.adjust_saturation, saturation_factor=saturation_factor),
                   clip_image=clip_image)(img)


def adjust_hue(img, hue_factor, clip_image=True):
    return Functor(pil_func=functools.partial(tf.adjust_hue, hue_factor=hue_factor),
                   clip_image=clip_image)(img)


def image_random_uniform_color(image, min_factor, max_factor, clip_image=False):
    def convert_torch(tensor):
        factor = min_factor + torch.rand((3, 1, 1)) * (max_factor - min_factor)
        return factor * tensor

    def convert_numpy(ndarray):
        factor = np.random.uniform(low=min_factor, high=max_factor, size=(1, 1, 3))
        return factor.astype(np.float32) * ndarray

    return Functor(torch_func=convert_torch,
                   numpy_func=convert_numpy,
                   clip_image=clip_image)(image)


def image_random_uniform_gamma(image, min_gamma, max_gamma, clip_image=False):
    def convert_tensor_or_array(tensor_or_ndarray):
        gamma = caffe_random_uniform(min_gamma, max_gamma)
        return tensor_or_ndarray ** gamma

    return Functor(torch_func=convert_tensor_or_array,
                   numpy_func=convert_tensor_or_array,
                   clip_image=clip_image)(image)


def image_random_gaussian_gamma(image, stddev=0.02, clip_image=False):
    def convert_tensor_or_array(tensor_or_ndarray):
        gamma = caffe_random_normal(0.0, stddev, exp=True)
        return tensor_or_ndarray ** gamma

    return Functor(torch_func=convert_tensor_or_array,
                   numpy_func=convert_tensor_or_array,
                   clip_image=clip_image)(image)


def image_random_gaussian_brightness(image, stddev=0.02, clip_image=False):
    def convert_tensor_or_array(tensor_or_ndarray):
        offset = caffe_random_uniform(0.0, stddev)
        return tensor_or_ndarray + offset

    return Functor(torch_func=convert_tensor_or_array,
                   numpy_func=convert_tensor_or_array,
                   clip_image=clip_image)(image)


def image_random_gaussian_contrast(image, stddev=0.02, clip_image=False):
    def convert_tensor_or_array(tensor_or_ndarray):
        contrast_factor = caffe_random_normal(0.0, stddev, exp=True)
        return (tensor_or_ndarray - 0.5) * contrast_factor + 0.5

    return Functor(torch_func=convert_tensor_or_array,
                   numpy_func=convert_tensor_or_array,
                   clip_image=clip_image)(image)


def image_random_gaussian_color(image, stddev=0.02, clip_image=False):
    def convert_torch(tensor):
        factor = caffe_random_normal(np.zeros((3, 1, 1)), stddev, exp=True)
        return factor * tensor

    def convert_numpy(ndarray):
        factor = caffe_random_normal(np.zeros((1, 1, 3)), stddev, exp=True)
        return factor * ndarray

    return Functor(torch_func=convert_torch,
                   numpy_func=convert_numpy,
                   clip_image=clip_image)(image)


def image_random_noise_from_normal(image, stddev=1.0, clip_image=True):
    def convert_torch(tensor, inner_stddev):
        mystddev = np.abs(caffe_random_normal(stddev=inner_stddev))
        noise = tensor.new_empty(tensor.size())
        noise.normal_(0.0, mystddev)
        tensor += noise
        return tensor

    def convert_numpy(ndarray, inner_stddev):
        mystddev = np.abs(caffe_random_normal(stddev=inner_stddev))
        noise = np.random.normal(scale=mystddev, size=ndarray.shape)
        ndarray += noise
        return ndarray

    return Functor(torch_func=functools.partial(convert_torch, inner_stddev=stddev),
                   numpy_func=functools.partial(convert_numpy, inner_stddev=stddev),
                   clip_image=clip_image)(image)


def image_random_uniform_noise(image, min_stddev, max_stddev, clip_image=False):
    def convert_torch(tensor, inner_min_stddev, inner_max_stddev):
        stddev = caffe_random_uniform(inner_min_stddev, inner_max_stddev)
        noise = tensor.new_empty(tensor.size())
        noise.normal_(0.0, stddev)
        tensor += noise
        return tensor

    def convert_numpy(ndarray, inner_min_stddev, inner_max_stddev):
        stddev = caffe_random_uniform(inner_min_stddev, inner_max_stddev)
        noise = np.random.normal(scale=stddev, size=ndarray.shape)
        ndarray += noise
        return ndarray

    return Functor(
        torch_func=functools.partial(convert_torch, inner_min_stddev=min_stddev, inner_max_stddev=max_stddev),
        numpy_func=functools.partial(convert_numpy, inner_min_stddev=min_stddev, inner_max_stddev=max_stddev),
        clip_image=clip_image)(image)
