from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use('Agg') # Only to save, not show
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import utils
import copy
import numba

# Import similarity measures
from sdtw_loss import SoftDTW

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])



def d_loop(netD, netG, optimizerD, optimizerG, criterion, image, scanpath, device, real_label, fake_label):
    
	# 1. Train D on real + fake
	optimizerD.zero_grad()


	d_real_decision = netD(image.to(device), scanpath.to(device), debug=False)
	b_size = len(scanpath)
	real_label_discriminator = torch.full((b_size,), real_label, dtype=torch.float, device=device)
	# Smooth labels
	sampl = np.random.uniform(low=0.85, high=1.15, size=(b_size,))
	for x in range(len(real_label_discriminator)):
		real_label_discriminator[x] = real_label_discriminator[x] * sampl[x]
	d_real_error = criterion(d_real_decision.view(-1), real_label_discriminator.view(-1)) 
	# Get accuracy
	acc_ok_real = d_real_decision[d_real_decision >= 0.5].numel()
	acc_bad_real = d_real_decision[d_real_decision < 0.5].numel()

	#  1B: Train D on fake
	# Generate batch of latent vectors
	noise = torch.randn(b_size, utils.random_z, 1, 1, device=device)
	noise = noise.squeeze()

	# Generate fake image batch with G
	with torch.no_grad():
		d_fake_data = netG(image.to(device), noise.to(device), batch_size=b_size, debug=True)
	d_fake_decision = netD(image.to(device), d_fake_data.detach())

	# Target label
	fake_label_discriminator = real_label_discriminator.clone()
	fake_label_discriminator.fill_(fake_label)
	# Smooth labels
	sampl = np.random.uniform(low=0, high=0.15, size=(b_size,))
	for x in range(len(real_label_discriminator)):
		fake_label_discriminator[x] = fake_label_discriminator[x] + sampl[x]
	d_fake_error = criterion(d_fake_decision.view(-1), fake_label_discriminator.view(-1))
	# Get accuracy
	acc_ok_fake = d_fake_decision[d_fake_decision <= 0.5].numel()
	acc_bad_fake = d_fake_decision[d_fake_decision > 0.5].numel()

	# Get total_accuracy
	acc_real = acc_ok_real / (acc_ok_real + acc_bad_real)
	acc_fake = acc_ok_fake / (acc_ok_fake + acc_bad_fake)

	# Loss
	d_loss = d_real_error + d_fake_error		# Penalize more fake errors, so it's forde to learn what is wrong
	d_loss.backward()
	optimizerD.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

	return [d_real_error.cpu().item(), d_fake_error.cpu().item()], [acc_real, acc_fake]


def g_loop(netD, netG, optimizerD, optimizerG, criterion, image, scanpath, device, real_label, fake_label):

	criterion_SoftDTW2 = SoftDTW(use_cuda=True, gamma=1.0, normalize=False)

	# 2. Train G on D's response (but DO NOT train D on these labels)
	optimizerG.zero_grad()
	optimizerD.zero_grad()

	b_size = len(scanpath)
	noise = torch.randn(b_size, utils.random_z, 1, 1, device=device)
	noise = noise.squeeze()

	if utils.unrolled_steps > 0:
		backup = copy.deepcopy(netD)
		for i in range(utils.unrolled_steps):
			d_unrolled_loop(netD, netG, optimizerD, optimizerG, criterion, image, scanpath, device, real_label, fake_label, noise=noise)

	g_fake_data = netG(image.to(device), noise, batch_size=b_size)
	dg_fake_decision = netD(image.to(device), g_fake_data)
	target = torch.ones_like(dg_fake_decision).to(device)

	# Smooth labels
	sampl = np.random.uniform(low=0.85, high=1.15, size=(b_size,))
	for x in range(len(target)):
		target[x] = target[x] * sampl[x]


	# BCE error
	bce_error = utils.lambda_BCE * criterion(dg_fake_decision.view(-1), target.view(-1)) 

	
	g_fake_data = torch.reshape(g_fake_data,(b_size, 2, g_fake_data.shape[-1] // 2))
	scanpath = torch.reshape(scanpath,(b_size, 2, scanpath.shape[-1] // 2))


	# DTW loss
	dtw_error = utils.lambda_DTW * torch.mean(criterion_SoftDTW2(g_fake_data.to(device), scanpath.to(device)))

	# Sum losses
	g_error = bce_error + dtw_error

	# Backward G loss
	g_error.backward()
	optimizerG.step()  # Only optimizes G's parameters

	if utils.unrolled_steps > 0:
		netD.load(backup)    
		del backup
	return g_error.cpu().item()

def d_unrolled_loop(netD, netG, optimizerD, optimizerG, criterion, image, scanpath, device, real_label, fake_label, noise=None):

	# 1. Train D on real + fake
	optimizerD.zero_grad()


	d_real_decision = netD(image.to(device), scanpath.to(device), debug=False)
	b_size = len(scanpath)
	real_label_discriminator = torch.full((b_size,), real_label, dtype=torch.float, device=device)
	# Smooth labels
	sampl = np.random.uniform(low=0.85, high=1.15, size=(b_size,))
	for x in range(len(real_label_discriminator)):
		real_label_discriminator[x] = real_label_discriminator[x] * sampl[x]
	d_real_error = criterion(d_real_decision.view(-1), real_label_discriminator.view(-1)) 

	#  1B: Train D on fake
	if noise is None:
		noise = torch.randn(b_size, utils.random_z, 1, 1, device=device)
		noise = noise.squeeze()

	# Generate fake image batch with G
	with torch.no_grad():
		d_fake_data = netG(image.to(device), noise.to(device), batch_size=b_size, debug=False)
	d_fake_decision = netD(image.to(device), d_fake_data)

	# Target label
	fake_label_discriminator = real_label_discriminator.clone()
	fake_label_discriminator.fill_(fake_label)
	# Smooth labels
	sampl = np.random.uniform(low=0, high=0.15, size=(b_size,))
	for x in range(len(real_label_discriminator)):
		fake_label_discriminator[x] = fake_label_discriminator[x] + sampl[x]
	d_fake_error = criterion(d_fake_decision.view(-1), fake_label_discriminator.view(-1))

	d_loss = d_real_error + d_fake_error		# Penalize more fake errors, so it's forde to learn what is wrong
	d_loss.backward(create_graph=True)
	optimizerD.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
	return d_real_error.cpu().item(), d_fake_error.cpu().item()


'''
Parameters:
	netG --> Generator Network
	netD --> Discriminator Network
	dataloader --> Dataloader from the training dataset
	criterion --> Criterion / Loss to be used
	fixed_noise --> random fixed noise to check the progress periodically
	real_label --> label for real data
	fake_label --> label for fake data
	optimizerD --> optimizer for the Discriminator
	optimizerG --> optimizer for the Generator
	device --> device to train in (cuda:0, cpu...)
'''

def train(netG, netD, dataloader, criterion, real_label, fake_label, optimizerD, optimizerG, device):

	# Lists to keep track of progress
	G_losses = []
	D_real_losses = []
	D_fake_losses = []
	D_real_acc = []
	D_fake_acc = []
	fake_scanpaths = [[], [], [], []]
	iters = 0


	torch.autograd.set_detect_anomaly(True)

	print("* Starting training loop")

	# For each epoch
	for epoch in range(utils.num_epochs):
		# For each batch in the dataloader
		for i, batch in enumerate(dataloader):

			image = batch[0]
			scanpath = batch[1]

			if (len(image) < 1) or (len(scanpath) < 1):
				continue

			image = torch.cat(image, dim=0)
			scanpath = torch.cat(scanpath, dim=0)

			d_infos = []
			acc_infos = []
			acc_real = 0
			acc_fake = 0
			for d_index in range(utils.d_steps):
				d_info, acc_info = d_loop(netD, netG, optimizerD, optimizerG, criterion, image, scanpath, device, real_label, fake_label)
				d_infos.append(d_info)
				acc_infos.append(acc_info)
			d_infos = np.mean(d_infos, 0)
			acc_infos = np.mean(acc_infos, 0)
			d_real_loss, d_fake_loss = d_infos
			acc_real, acc_fake = acc_infos

			g_infos = []
			for g_index in range(utils.g_steps):
				g_info = g_loop(netD, netG, optimizerD, optimizerG, criterion, image, scanpath, device, real_label, fake_label)
				g_infos.append(g_info)
			g_infos = np.mean(g_infos)
			g_loss = g_infos

			############################
			# Show status
			############################

			# Output training stats
			if i % 500 == 0:
				print('Progress --> [%d/%d][%d/%d]\tLoss_D_real: %.4f\tLoss_D_fake: %.4f\tLoss_G: %.4f'
					  % (epoch, utils.num_epochs, i, len(dataloader),
						 d_real_loss,d_fake_loss, g_loss))

				# Save Losses for plotting later
				G_losses.append(g_loss)
				D_fake_losses.append(d_fake_loss)
				D_real_losses.append(d_real_loss)
				# Save accuracy for plotting later
				D_fake_acc.append(acc_fake)
				D_real_acc.append(acc_real)

		# Checkpointing (each iteraton)
		weightsG = copy.deepcopy(netG.state_dict())
		weightsD = copy.deepcopy(netD.state_dict())

		# Save models
		torch.save(netG, utils.model_folder + 'generator_' + str(epoch) + '.pth')
		torch.save(netD, utils.model_folder + 'discriminator_' + str(epoch) + '.pth')



	





