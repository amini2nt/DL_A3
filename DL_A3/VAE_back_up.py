import numpy as np
import ipdb
import os
import torch
import torch.utils.data as data_utils
from scipy.io import loadmat
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

class VAE(nn.Module):
	def __init__(self, image_size=784, z_dim=100):
		super(VAE, self).__init__()

		self.encoder = nn.Sequential(
						nn.Conv2d(1,32, kernel_size=(3,3)),
						nn.ELU(),
						nn.AvgPool2d(kernel_size=2, stride=2),
						nn.Conv2d(32,64,kernel_size=(3,3)),
						nn.ELU(),
						nn.AvgPool2d(kernel_size=2, stride=2),
						nn.Conv2d(64, 256, kernel_size=(5,5)),
						nn.ELU(),
						nn.Linear(in_features=256, out_features=(100 * 2))
						)

		self.decoder = nn.Sequential(
						nn.Linear(in_features = 100, out_features = 256),
						nn.ELU(),
						nn.Conv2d(256, 64, kernel_size=(5,5), padding=(4,4)),
						nn.ELU(),
						nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
						nn.ELU(),
						nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
						nn.ELU(),
						nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2))
						)

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		z = mu + std * esp
		return z
	
	def forward(self, x):
		h = self.encoder(x)
		mu, logvar = torch.chunk(256, 2, dim=1)
		z = self.reparameterize(mu, logvar)
		return self.decoder(z), mu, logvar


def loss_fn(recon_x, x, mu, logvar):
	BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
	KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
	return BCE + KLD

def load_static_mnist(batch_size = 32, test_batch_size=32):
	input_size = [1, 28, 28]
	input_type = 'binary'
	dynamic_binarization = False
	def lines_to_np_array(lines):
		return np.array([[int(i) for i in line.split()] for line in lines])
	with open(os.path.join('mnist_static', 'binarized_mnist_train.amat')) as f:
		lines = f.readlines()
	x_train = lines_to_np_array(lines).astype('float32')
	with open(os.path.join('mnist_static', 'binarized_mnist_valid.amat')) as f:
		lines = f.readlines()
	x_val = lines_to_np_array(lines).astype('float32')
	with open(os.path.join('mnist_static', 'binarized_mnist_test.amat')) as f:
		lines = f.readlines()
	x_test = lines_to_np_array(lines).astype('float32')
	np.random.shuffle(x_train)
	y_train = np.zeros( (x_train.shape[0], 1) )
	y_val = np.zeros( (x_val.shape[0], 1) )
	y_test = np.zeros( (x_test.shape[0], 1) )
	train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
	validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
	val_loader = data_utils.DataLoader(validation, batch_size=test_batch_size, shuffle=False)
	test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
	test_loader = data_utils.DataLoader(test, batch_size=test_batch_size, shuffle=True)
	return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = load_static_mnist()
fixed_x, _ = next(iter(train_loader))
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4 , betas= [0.9, 0.999])
number_of_epochs = 20
for i in range(100000):
	recon_images, mu, logvar = model(fixed_x)
	loss = loss_fn(recon_images, fixed_x, mu, logvar)
	print(loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
ipdb.set_trace()
