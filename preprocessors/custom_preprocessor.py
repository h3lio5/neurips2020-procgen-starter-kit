#!/usr/bin/env python
import random
import numpy as np
from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog
from torchvision.transforms import ColorJitter


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop"""
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3)
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def random_crop(imgs, size=64, w1=None, h1=None, return_w1_h1=False):
    """Vectorized random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
        'must either specify both w1 and h1 or neither of them'

    is_tensor = isinstance(imgs, torch.Tensor)
    if is_tensor:
        assert imgs.is_cuda, 'input images are tensors but not cuda!'
        return random_crop_cuda(imgs,
                                size=size,
                                w1=w1,
                                h1=h1,
                                return_w1_h1=return_w1_h1)

    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return imgs, None, None
        return imgs

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    if w1 is None:
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)

    windows = view_as_windows(imgs, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[np.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


class MyPreprocessorClass(Preprocessor):
    """Custom preprocessing for observations

    Adopted from https://docs.ray.io/en/master/rllib-models.html#custom-preprocessors
    """
    def _init_shape(self, obs_space, options):
        return (64, 64, 3)  # New shape after preprocessing

    def transform(self, observation):
        # random cropping
        # h, w = observation.shape[:2]
        # new_h, new_w = 56, 56
        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)
        # observation = observation[top:top + new_h, left:left + new_w]
        observation = observation.permute(2, 0, 1)
        transform_module = ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4,
                                       hue=0.5)
        observation = transform_module(observation).permute(1, 2, 0)

        return observation


ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)