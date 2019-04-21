from comet_ml import Experiment

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
	def __init__(self):
		super(VAE, self).__init__()

		self.encoder_conv = nn.Sequential(
						nn.Conv2d(1,32, kernel_size=(3,3)),
						nn.ELU(),
						nn.AvgPool2d(kernel_size=2, stride=2),
						nn.Conv2d(32,64,kernel_size=(3,3)),
						nn.ELU(),
						nn.AvgPool2d(kernel_size=2, stride=2),
						nn.Conv2d(64, 256, kernel_size=(5,5)),
						nn.ELU(),
						)
		self.encoder_fc =	nn.Linear(in_features=256, out_features=(100 * 2))

		self.decoder_fc = nn.Sequential(
					nn.Linear(in_features = 100, out_features = 256),
					nn.ELU()
					)



		self.decoder_conv = nn.Sequential(
						nn.Conv2d(256, 64, kernel_size=(5,5), padding=(4,4)),
						nn.ELU(),
						nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
						nn.ELU(),
						nn.Upsample(scale_factor=2, mode='bilinear'),
						nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
						nn.ELU(),
						nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2))
						#nn.Sigmoid()
						)

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		z = mu + std * esp
		return z
	def decoder(self, z):
		z = self.decoder_fc(z)
		z = z.unsqueeze(2).unsqueeze(3)
		x_hat = self.decoder_conv(z)
		return x_hat
	
	def forward(self, x):
		#ipdb.set_trace()
		h = self.encoder_conv(x)
		h = h.view(h.shape[0], h.shape[1])
		h = self.encoder_fc(h)
		mu, logvar = torch.chunk(h, 2, dim=1)
		z = self.reparameterize(mu, logvar)
		z = self.decoder_fc(z)
		z = z.unsqueeze(2).unsqueeze(3)
		return self.decoder_conv(z), mu, logvar

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

def compute_prob(model, data_loader, K):

	
	batchNo = 0
	final_log = 0.0
	for batch_idx, (x, _) in enumerate(data_loader):
		batchNo += 1
		x = Variable(x).view(x.shape[0],1,28,28)
		_, mus, logvars = model(x)
		var = logvars.exp_()

		## put for loop for K times
		
		probs = torch.zeros(x.shape[0])
		for k in range(K):
			z = model.reparameterize(mus, logvars)


			for i in range(x.shape[0]):
				qd = torch.distributions.multivariate_normal.MultivariateNormal(mus[i],var[i].diag())
				q = qd.log_prob(z[i])
				pd = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(100), torch.eye(100))
				p = pd.log_prob(z[i])
				p_x = torch.sigmoid(model.decoder(z[i].unsqueeze(0)).flatten())
				pd = torch.distributions.bernoulli.Bernoulli(p_x)
				m = pd.log_prob(x[0].flatten()).sum()
				
				log_prob = m + p - q
				prob = log_prob.exp()
				probs[i] += prob

		probs = probs/K
		logprobs = torch.log(probs)
		final_log += logprobs.sum()
		print(logprobs.sum())
	final_log /= batchNo
	return final_log




train_loader, val_loader, test_loader = load_static_mnist()
model = VAE()
PATH = "best_model_from_" + str(0) + "_epoch"
model.load_state_dict(torch.load(PATH))
model.eval()

K = 200
val_log_likelihood = compute_prob(model, val_loader, K)

test_log_likelihood = compute_prob(model, test_loader, K)

print(val_log_likelihood)
print(test_log_likelihood)

