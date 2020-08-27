#!/usr/bin/env python
import random
import numpy as np
from ray.rllib.utils import try_import_torch
torch, nn = try_import_torch()
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog
from torchvision.transforms import ColorJitter
from PIL import Image


class MyPreprocessorClass(Preprocessor):
    """Custom preprocessing for observations

    Adopted from https://docs.ray.io/en/master/rllib-models.html#custom-preprocessors
    """
    def _init_shape(self, obs_space, options):
        if self.gray_scale:
            return (64, 64, 1)
        return (64, 64, 3)  # New shape after preprocessing

    def transform(self, observation):

        flag = np.random.randint(1, 6)
        if flag == 1:
            observation = Image.fromarray(observation.astype('uint8'), 'RGB')
            transform_module = ColorJitter(brightness=0.4,
                                           contrast=0.4,
                                           saturation=0.4,
                                           hue=0.5)
            return np.array(transform_module(observation))

        elif flag == 2:
            return np.flip(observation, 1)

        elif flag == 3:
            h1 = np.random.randint(10, 20)
            w1 = np.random.randint(10, 20)
            observation[h1:h1 + h1, w1:w1 + w1, :] = 0
            return observation

        elif flag == 4:
            h1 = np.random.randint(10, 20)
            w1 = np.random.randint(10, 20)
            rand_color = np.random.randint(0, 255, size=(3, )) / 255.
            observation[h1:h1 + h1, w1:w1 + w1, :] = np.tile(
                rand_color.reshape(1, 1, -1),
                observation[h1:h1 + h1, w1:w1 + w1, :].shape[:2] + (1, ))
            return observation

        elif flag == 5:
            self.gray_scale = True
            observation = observation[:, :,
                                      0] * 0.2989 + observation[:, :,
                                                                1] * 0.587 + observation[:, :,
                                                                                         2] * 0.114

            return np.expand_dims(observation, axis=2)


ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
