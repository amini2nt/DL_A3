#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:49:36 2019

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

sample_dir = "GAN/samples"

MODEL = "GAN"
generator = load_model("gan_generator.h5")

for i in range(1000):
    if MODEL == "GAN":
        sample = np.random.rand(1, 100)
        sample = generator.predict(sample)
        sample = (sample * 127.5) + 127.5
    else:
        sample = np.random.normal(size=(1, 100))
        sample = generator.predict(sample)
        sample = sample * 255
    sample = np.squeeze(np.round(sample).astype(np.uint8))
    output_image = Image.fromarray(sample, mode="RGB")
    output_image.save(os.path.join(sample_dir, 'sample' + str(i) + '.png'))
