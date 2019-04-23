'''Example of VAE on MNIST dataset using CNN
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import ReLU, LeakyReLU, ELU, BatchNormalization
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
import argparse
import os
import scipy.io

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! '
          'Please install it (e.g. with pip install pillow)')
    exit()


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 4
    image_list = [image_stack[i, :, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images

def generate_images(generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    test_image_stack = generator_model.predict(np.random.normal(size=(10, 100)))
    test_image_stack = (test_image_stack * 255)
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output)
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

output_dir = "samples"

## MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# First we load the image data, reshape it and normalize it to the range [-1, 1]
data_train = scipy.io.loadmat('train_32x32.mat')
data_test = scipy.io.loadmat('train_32x32.mat')

(x_train, y_train), (x_test, y_test) = (data_train["X"], data_train["y"]), (data_test["X"], data_test["y"])

image_size = x_train.shape[1]
#X_train = np.concatenate((X_train, X_test), axis=-1)

x_train = np.rollaxis(x_train, 3)
x_test = np.rollaxis(x_test, 3)
if K.image_data_format() == 'channels_first':
    x_train = np.rollaxis(x_train, 3, 1)
    x_test = np.rollaxis(x_test, 3, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 3)
batch_size = 64
kernel_size = 5
latent_dim = 100
epochs = 100

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(filters=64, kernel_size=kernel_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(filters=64, kernel_size=kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(filters=128, kernel_size=kernel_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = latent_inputs
x = Dense(1024)(x)
x = LeakyReLU()(x)
x = Dense(shape[1] * shape[2] * shape[3])(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=64, kernel_size=kernel_size, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=64, kernel_size=kernel_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(0.0001))
    vae.summary()
#    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        for epoch in range(300):
            # train the autoencoder
            vae.fit(x_train,
                    epochs=1,
                    batch_size=batch_size,
                    validation_split=0.2)
            vae.save_weights('vae.h5')
    
            generate_images(decoder, output_dir, epoch)
            decoder.save("vae_decoder" + str(epoch) + ".h5")
