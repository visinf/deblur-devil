# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import functools
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms as vision_transforms

from datasets import common
from datasets import factory
from datasets.my_transforms import transforms
from utils import system


class ExampleData:
    input_filenames = None
    gt_filenames = None
    basenames = None


def clip(value, amin, amax):
    return min(amax, max(value, amin))


class GoPro(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 gamma_corrected_data=False,
                 dstype='train',
                 random_brightness=False,
                 random_channel_permutation=False,
                 random_contrast=False,
                 random_crop=(-1, -1),
                 random_flips=False,
                 random_hue=False,
                 random_multiplicative_colors=False,
                 random_noise=False,
                 random_reverse=False,
                 random_rotate=False,
                 random_saturation=False,
                 random_scales=None,
                 random_scales_min=0.25,
                 sequence_length=5,
                 num_samples_per_example=1,
                 num_examples=-1):

        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise ValueError('Dataset root directory \'%s\' not found!')
        if dstype not in ['train', 'test', 'full']:
            raise ValueError("Unknown dstype '{}'".format(dstype))

        self.args = args
        self.has_random_reverse = random_reverse
        self.num_samples_per_example = num_samples_per_example
        self.reference_index = sequence_length // 2
        self.sequence_length = sequence_length

        # ----------------------------------------------------------
        # Read in examples
        # ----------------------------------------------------------
        self.examples = self.read_examples(
            root,
            dstype=dstype,
            gamma_corrected_data=gamma_corrected_data,
            sequence_length=sequence_length)
        if len(self.examples) == 0:
            raise ValueError("Could not find any examples")

        # ----------------------------------------------------------
        # image conversion transforms
        # ----------------------------------------------------------
        self.to_pil = transforms.TransformChainer([vision_transforms.ToPILImage()])
        self.to_tensor = transforms.TransformChainer([vision_transforms.transforms.ToTensor()])

        # ----------------------------------------------------------
        # Random crops
        # ----------------------------------------------------------
        self.random_resized_crop_transform = None
        if any([x > 0 for x in random_crop]):
            if random_scales is None or random_scales == 'none':
                self.random_resized_crop_transform = transforms.MultiImageRandomResizedCrop(
                    size=random_crop, mode='none')
            elif random_scales == 'weak':
                self.random_resized_crop_transform = transforms.MultiImageRandomResizedCrop(
                    size=random_crop, scale_choices=(1 / 4, 1 / 3, 1 / 2))
            elif random_scales == 'strong':
                self.random_resized_crop_transform = transforms.MultiImageRandomResizedCrop(
                    size=random_crop, mode='random_range', scale_range=(random_scales_min, 1.0))
            else:
                raise ValueError('Unknown random_scales')

        # ----------------------------------------------------------
        # Random rotations
        # ----------------------------------------------------------
        self.random_rotation = None
        if random_rotate:
            self.random_rotation = transforms.RandomChoice(
                [transforms.MultiImageRandomRotation(degrees=(0, 0)),
                 transforms.MultiImageRandomRotation(degrees=(90, 90)),
                 transforms.MultiImageRandomRotation(degrees=(180, 180)),
                 transforms.MultiImageRandomRotation(degrees=(270, 270))])

        # ----------------------------------------------------------
        # Random flips
        # ----------------------------------------------------------
        self.random_hflip = None
        self.random_vflip = None
        if random_flips:
            self.random_hflip = transforms.MultiImageRandomHorizontalFlip()
            self.random_vflip = transforms.MultiImageRandomVerticalFlip()

        # ----------------------------------------------------------
        # Random multiplicative colors
        # ----------------------------------------------------------
        self.random_multiplicative_colors = None
        if random_multiplicative_colors:
            self.random_multiplicative_colors = transforms.ConcatTransformSplitChainer([
                transforms.RandomMultiplicativeColor(
                    min_factor=0.5, max_factor=2.0, clip_image=True)])

        # ----------------------------------------------------------
        # Random photometrics
        # ----------------------------------------------------------
        self.photometric_transform = None
        if any([random_brightness, random_contrast, random_hue, random_saturation]):
            brightness = 0.5 if random_brightness else 0
            contrast = 0.5 if random_contrast else 0
            saturation = 0.5 if random_saturation else 0
            hue = 0.5 if random_hue else 0
            self.photometric_transform = transforms.ConcatTransformSplitChainer([
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue),
            ])

        # vision_transforms.ToTensor(),

        # ----------------------------------------------------------
        # Random noise
        # ----------------------------------------------------------
        self.random_noise_transform = None
        if random_noise:
            self.random_noise_transform = transforms.ConcatTransformSplitChainer([
                transforms.RandomNoise(min_stddev=0.0, max_stddev=0.04, clip_image=True)
            ])

        # ----------------------------------------------------------
        # Random channel permutation
        # ----------------------------------------------------------
        self.random_channel_permutation = None
        if random_channel_permutation:
            self.random_channel_permutation = transforms.ConcatTransformSplitChainer([
                transforms.RandomChannelPermutation()
            ])

        # ----------------------------------------------------------
        # Restrict dataset indices if num_examples is given
        # ----------------------------------------------------------
        if num_examples > 0:
            restricted_indices = common.deterministic_indices(
                seed=0, k=num_examples, n=len(self.examples))
            self.examples = [self.examples[i] for i in restricted_indices]

        self.size = len(self.examples)

    @staticmethod
    def read_examples(root, dstype, gamma_corrected_data, sequence_length):
        # ----------------
        # Read subfolders
        # ----------------
        subdirs = []
        if dstype in ['train', 'test']:
            path = os.path.join(root, dstype)
            subdirs = sorted(system.get_subdirs(path, level=0))
        elif dstype == 'full':
            train_path = os.path.join(root, 'train')
            train_subdirs = sorted(system.get_subdirs(train_path, level=0))
            test_path = os.path.join(root, 'test')
            test_subdirs = sorted(system.get_subdirs(test_path, level=0))
            subdirs = train_subdirs + test_subdirs
        logging.info('num_sequences: %i' % len(subdirs))
        examples = []
        for subdir in subdirs:
            basename = system.fileparts(subdir)[1]
            inputdir = os.path.join(subdir, 'blur_gamma') \
                if gamma_corrected_data else os.path.join(subdir, 'blur')
            gtdir = os.path.join(subdir, 'sharp')

            if not os.path.isdir(inputdir):
                raise ValueError('Could not find blur directory %s', inputdir)
            if not os.path.isdir(gtdir):
                raise ValueError('Could not find sharp directory %s', gtdir)

            input_filenames = sorted(glob(os.path.join(inputdir, "*.png")))
            gt_filenames = sorted(glob(os.path.join(gtdir, '*.png')))

            assert len(input_filenames) == len(gt_filenames), "Number of files is inconsistent"

            # add examples for this subdir
            num_examples_for_subdir = len(input_filenames)

            sequence_rad = sequence_length // 2

            for i in range(num_examples_for_subdir):
                example = ExampleData()

                example.input_filenames = []
                for j in range(i - sequence_rad, i + sequence_rad + 1):
                    j = clip(j, 0, num_examples_for_subdir - 1)
                    example.input_filenames.append(input_filenames[j])

                example.gt_filenames = []
                for j in range(i - sequence_rad, i + sequence_rad + 1):
                    j = clip(j, 0, num_examples_for_subdir - 1)
                    example.gt_filenames.append(gt_filenames[j])

                # another sanity check testing input == gt ordering
                for input_filename, gt_filename in zip(example.input_filenames,
                                                       example.gt_filenames):
                    base1 = system.fileparts(input_filename)[1]
                    base2 = system.fileparts(gt_filename)[1]
                    assert (base1 == base2), "Example basenames are inconsistent"
                example.basenames = [
                    os.path.join(basename, system.fileparts(x)[1]) for x in example.input_filenames
                ]
                examples.append(example)
        return examples

    def read_cached(self, input_filenames, gt_filename):
        numpy_filename = gt_filename.replace('.png', '-seq%i.npy' % self.sequence_length)
        if not os.path.isfile(numpy_filename):
            inputs_np = [common.read_image_as_byte(fn) for fn in input_filenames]
            gt_np = common.read_image_as_byte(gt_filename)
            images = inputs_np
            images += [gt_np]
            ndarray = np.stack(images, axis=0)
            np.save(numpy_filename, ndarray)
        else:
            ndarray = np.load(numpy_filename)
            inputs_np = np.split(ndarray[0:self.sequence_length, :, :, :], self.sequence_length, axis=0)
            inputs_np = [np.squeeze(arr, axis=0) for arr in inputs_np]
            gt_np = ndarray[-1, :, :, :]

        if gt_np.dtype == np.float32:
            inputs_np = [(x * 255).astype(np.uint8) for x in inputs_np]
            gt_np = (gt_np * 255).astype(np.uint8)
        return inputs_np, gt_np

    # -------------------------------------
    # Actual example retrieval
    # -------------------------------------
    def __getitem__(self, index):
        # -------------------------------------
        # Meta details and index
        # -------------------------------------
        index = index % self.size
        example = self.examples[index]
        input_filenames = example.input_filenames
        gt_filename = example.gt_filenames[self.reference_index]
        basename = example.basenames[self.reference_index]

        # --------------------------------------------
        # Read images from files into numpy ndarrays
        # --------------------------------------------
        inputs_np, gt_np = self.read_cached(input_filenames, gt_filename)

        # --------------------------------------------
        # Random reverse of sequence
        # --------------------------------------------
        if self.has_random_reverse:
            if common.coin_flip():
                inputs_np = list(reversed(inputs_np))

        # --------------------------------------------
        # Concatenate inputs and gt for co transforms
        # We kinda have to remember: GT = LAST
        # --------------------------------------------
        images_np = inputs_np + [gt_np]

        # --------------------------------------------
        # Initially convert ndarrays into PIL images
        # --------------------------------------------
        raw_images_pil = self.to_pil(*images_np)

        # --------------------------------------------
        # Augmentation loop
        # --------------------------------------------
        inputs = []
        gt = []
        basenames = []
        indices = torch.zeros(self.num_samples_per_example)
        reference_frames = torch.zeros(self.num_samples_per_example)

        for j in range(self.num_samples_per_example):
            basenames.append(basename)
            indices[j] = index
            reference_frames[j] = self.reference_index

            # --------------------------------------------
            # ALways start from initial images
            # --------------------------------------------
            images_pil = raw_images_pil

            # --------------------------------------------
            # Apply random stuff
            # --------------------------------------------
            if self.random_resized_crop_transform is not None:
                images_pil = self.random_resized_crop_transform(*images_pil)
            if self.random_multiplicative_colors is not None:
                images_pil = self.random_multiplicative_colors(*images_pil)
            if self.random_hflip is not None:
                images_pil = self.random_hflip(*images_pil)
            if self.random_vflip is not None:
                images_pil = self.random_vflip(*images_pil)
            if self.random_rotation is not None:
                images_pil = self.random_rotation(*images_pil)

            # --------------------------------------------
            # Convert to torch tensors
            # --------------------------------------------
            image_tensors = self.to_tensor(*images_pil)

            if self.photometric_transform is not None:
                image_tensors = self.photometric_transform(*image_tensors)

            # --------------------------------------------
            # Convert to torch tensors
            # --------------------------------------------
            # image_tensors = self.to_tensor(*images_pil)

            # ----------------------------------------------------------
            # Random channel permutation
            # ----------------------------------------------------------
            if self.random_channel_permutation is not None:
                image_tensors = self.random_channel_permutation(*image_tensors)

            # --------------------------------------------
            # Split all images again into inputs/gt
            # --------------------------------------------
            input_tensors = image_tensors[0:-1]
            gt_tensor = image_tensors[-1]

            # --------------------------------------------
            # Apply random noise only to inputs
            # --------------------------------------------
            if self.random_noise_transform is not None:
                input_tensors = self.random_noise_transform(*input_tensors)

            # --------------------------------------------
            # Append inputs and gt tensors
            # --------------------------------------------
            inputs.append(input_tensors)
            gt.append(gt_tensor)

        gt = torch.stack(gt, dim=0)

        example_dict = {
            "basename": basenames,
            "index": indices,
            "reference_index": reference_frames,
            "target1": gt
        }
        for j in range(self.sequence_length):
            inputs_j = [inputs[i][j] for i in range(self.num_samples_per_example)]
            example_dict["input%i" % (j + 1)] = torch.stack(inputs_j, dim=0)

        return example_dict

    def __len__(self):
        return self.size


class GoProNahTrain(GoPro):
    def __init__(self,
                 args,
                 root,
                 gamma_corrected_data=False,
                 random_brightness=False,
                 random_channel_permutation=False,
                 random_contrast=False,
                 random_crop=(-1, -1),
                 random_flips=True,
                 random_hue=False,
                 random_multiplicative_colors=False,
                 random_noise=False,
                 random_reverse=True,
                 random_rotate=True,
                 random_saturation=False,
                 random_scales='none',
                 random_scales_min=0.25,
                 sequence_length=5,
                 num_samples_per_example=8,
                 num_workers=12,
                 num_examples=-1):
        self.num_workers = num_workers
        super().__init__(
            args,
            root=root,
            gamma_corrected_data=gamma_corrected_data,
            dstype='train',
            random_brightness=random_brightness,
            random_channel_permutation=random_channel_permutation,
            random_contrast=random_contrast,
            random_crop=random_crop,
            random_flips=random_flips,
            random_hue=random_hue,
            random_multiplicative_colors=random_multiplicative_colors,
            random_noise=random_noise,
            random_reverse=random_reverse,
            random_rotate=random_rotate,
            random_saturation=random_saturation,
            random_scales=random_scales,
            random_scales_min=random_scales_min,
            sequence_length=sequence_length,
            num_samples_per_example=num_samples_per_example,
            num_examples=num_examples)


class GoProNahTest(GoPro):
    def __init__(self,
                 args,
                 root,
                 gamma_corrected_data=False,
                 random_brightness=False,
                 random_channel_permutation=False,
                 random_contrast=False,
                 random_crop=(-1, -1),
                 random_flips=False,
                 random_hue=False,
                 random_multiplicative_colors=False,
                 random_noise=False,
                 random_reverse=False,
                 random_rotate=False,
                 random_saturation=False,
                 random_scales=None,
                 random_scales_min=1.0,
                 sequence_length=5,
                 num_samples_per_example=1,
                 num_workers=12,
                 num_examples=-1):
        self.num_workers = num_workers
        super().__init__(
            args,
            root=root,
            gamma_corrected_data=gamma_corrected_data,
            dstype='test',
            random_brightness=random_brightness,
            random_channel_permutation=random_channel_permutation,
            random_contrast=random_contrast,
            random_crop=random_crop,
            random_flips=random_flips,
            random_hue=random_hue,
            random_multiplicative_colors=random_multiplicative_colors,
            random_noise=random_noise,
            random_reverse=random_reverse,
            random_rotate=random_rotate,
            random_saturation=random_saturation,
            random_scales=random_scales,
            random_scales_min=random_scales_min,
            sequence_length=sequence_length,
            num_samples_per_example=num_samples_per_example,
            num_examples=num_examples)


class GoProNahFull(GoPro):
    def __init__(self,
                 args,
                 root,
                 gamma_corrected_data=False,
                 random_brightness=True,
                 random_channel_permutation=False,
                 random_contrast=True,
                 random_crop=(-1, -1),
                 random_flips=True,
                 random_hue=True,
                 random_multiplicative_colors=False,
                 random_noise=False,
                 random_reverse=True,
                 random_rotate=True,
                 random_saturation=False,
                 random_scales='none',
                 random_scales_min=0.25,
                 sequence_length=5,
                 num_samples_per_example=8,
                 num_workers=12,
                 num_examples=-1):
        self.num_workers = num_workers
        super().__init__(
            args,
            root=root,
            gamma_corrected_data=gamma_corrected_data,
            dstype='full',
            random_brightness=random_brightness,
            random_channel_permutation=random_channel_permutation,
            random_contrast=random_contrast,
            random_crop=random_crop,
            random_flips=random_flips,
            random_hue=random_hue,
            random_multiplicative_colors=random_multiplicative_colors,
            random_noise=random_noise,
            random_reverse=random_reverse,
            random_rotate=random_rotate,
            random_saturation=random_saturation,
            random_scales=random_scales,
            random_scales_min=random_scales_min,
            sequence_length=sequence_length,
            num_samples_per_example=num_samples_per_example,
            num_examples=num_examples)


factory.register("GoProNahTrain", GoProNahTrain)
factory.register("GoProNahTest", GoProNahTest)
factory.register("GoProNahFull", GoProNahFull)

GoProNahGammaTrain = functools.partial(GoProNahTrain, gamma_corrected_data=True)
GoProNahGammaTest = functools.partial(GoProNahTest, gamma_corrected_data=True)
GoProNahGammaFull = functools.partial(GoProNahFull, gamma_corrected_data=True)

factory.register("GoProNahGammaTrain", GoProNahGammaTrain)
factory.register("GoProNahGammaTest", GoProNahGammaTest)
factory.register("GoProNahGammaFull", GoProNahGammaFull)
