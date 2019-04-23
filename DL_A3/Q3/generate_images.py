#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:33:40 2019

@author: zeyu
"""

import os
import numpy as np
from keras.models import load_model

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! '
          'Please install it (e.g. with pip install pillow)')
    exit()

sample_dir = "samples"

MODEL = "GAN"
generator = load_model("gan_generator.h5")

def get_image_from_sample(sample, row_size=10):
    if MODEL == "GAN":
        sample = (sample * 127.5) + 127.5
    else:
        sample = sample * 255
    sample = np.squeeze(np.round(sample).astype(np.uint8))
    sample = np.concatenate([np.concatenate([sample[i * row_size + j, :, :, :] for j in range(row_size)], axis=1) for i in range(10)], axis=0)
    output_image = Image.fromarray(sample, mode="RGB")
    return output_image

# random samples
if MODEL == "GAN":
    sample = generator.predict(np.random.rand(100, 100))
else:
    sample = generator.predict(np.random.normal(size=(100, 100)))
output_image = get_image_from_sample(sample)
output_image.save(os.path.join(sample_dir, 'random.png'))

# perturbation on random sample
if MODEL == "GAN":
    original_distribution = np.random.rand(1, 100)
else:
    original_distribution = np.random.normal(size=(1, 100))

sample = generator.predict(original_distribution)
if MODEL == "GAN":
    sample = (sample * 127.5) + 127.5
else:
    sample = sample * 255
sample = np.squeeze(np.round(sample).astype(np.uint8))
output_image = Image.fromarray(sample, mode="RGB")
output_image.save(os.path.join(sample_dir, 'perturbation_original.png'))

#for i in range(100):
perturbation = np.eye(100) * 1
sample = generator.predict(original_distribution + perturbation)
output_image = get_image_from_sample(sample)
output_image.save(os.path.join(sample_dir, 'perturbation.png'))

# interpolation between 2 latent distributions
if MODEL == "GAN":
    distribution1 = np.random.rand(10, 100)
    distribution2 = np.random.rand(10, 100)
else:
    distribution1 = np.random.normal(size=(10, 100))
    distribution2 = np.random.normal(size=(10, 100))

distribution = np.zeros((11, 10, 100))

for i in range(11):
    a = i / 10.
    distribution[i] = a * distribution1 + (1 - a) * distribution2

distribution = np.rollaxis(distribution, 1).reshape(110, 100)
sample = generator.predict(distribution)
output_image = get_image_from_sample(sample, 11)
output_image.save(os.path.join(sample_dir, 'interp_distribution.png'))

# interpolation between 2 generated samples
sample1 = generator.predict(distribution1)
sample2 = generator.predict(distribution2)

sample = np.zeros((11, 10, sample1.shape[1], sample1.shape[2], sample1.shape[3]))

for i in range(11):
    a = i / 10.
    sample[i] = a * sample1 + (1 - a) * sample2

sample = np.rollaxis(sample, 1).reshape(110, sample1.shape[1], sample1.shape[2], sample1.shape[3])
output_image = get_image_from_sample(sample, 11)
output_image.save(os.path.join(sample_dir, 'interp_sample.png'))
