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


def loss_fn(recon_x, x, mu, logvar):
	recon_x = recon_x.view(recon_x.shape[0], -1)
	x = x.view(x.shape[0], -1)

	BCE = nn.BCEWithLogitsLoss(reduction='sum')(recon_x.float(), x.float())
	KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
	return (BCE + KLD)/x.shape[0]

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

def train(model, epoch, stepTrain, train_loader):
	model.train()
	experiment.train()
	experiment.log_current_epoch(epoch)
	allEpochTrainingLoss = []

	for batch_idx, (x, _) in enumerate(train_loader):
		optimizer.zero_grad()
		x= Variable(x).view(x.shape[0],1,28,28)
		recon_images, mus, logvars = model(x)
		loss = loss_fn(recon_images, x, mus, logvars)
		loss.backward()
		print(loss.item())
		optimizer.step()
		experiment.log_metric("loss", loss.item(), step= stepTrain)
		allEpochTrainingLoss.append(loss.item())
		stepTrain += 1
	return stepTrain, allEpochTrainingLoss


def eval(model, epoch, val_loader):
	model.eval()
	experiment.validate()
	experiment.log_current_epoch(epoch)
	eval_loss = 0
	batchNo = 0
	for batch_idx, (x, _) in enumerate(val_loader):
		x = Variable(x).view(x.shape[0],1,28,28)
		recon_images, mus, logvars = model(x)
		eval_loss += loss_fn(recon_images, x, mus, logvars).item()
		batchNo +=1

	eval_loss /=  batchNo
	experiment.log_metric("Evaluationloss", eval_loss, step=epoch)

	return eval_loss








bestEpoch = 0
experiment = Experiment(api_key="V1ZVYej7DxAyiXRVoeAs4JWZb",
                        project_name="VAE", workspace="amini2nt")
number_of_epochs = 20
allEpochsTrainingLoss = []
allEpochsEvaluationLoss  = []
train_loader, val_loader, test_loader = load_static_mnist()
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4 , betas= [0.9, 0.999])
bestLoss = float("inf")

stepTrain = 0
for epoch in range(50):
	stepTrain, allEpochTrainingLoss = train(model, epoch, stepTrain, train_loader)
	eval_loss = eval(model, epoch, val_loader)
	if eval_loss < bestLoss:
		bestLoss = eval_loss
		print(bestLoss," is best loss so far which belongs to ", epoch)
		bestEpoch = epoch
		directory = "best_model_50.p"
		bestModel = model.state_dict()
		torch.save(bestModel, directory) 

	allEpochsTrainingLoss.extend(allEpochTrainingLoss)
	allEpochsEvaluationLoss.append(eval_loss)
pickle.dump( allEpochsEvaluationLoss, open( "allEpochsEvaluationLoss.p", "wb" ) )
pickle.dump( allEpochsTrainingLoss, open( "allEpochTrainingLoss.p", "wb" ) )


