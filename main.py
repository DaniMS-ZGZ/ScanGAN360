from __future__ import print_function
#%matplotlib inline
import configargparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

'''
Suppress SourceChangeWarning - we have removed comment lines and debug options from the source code.
'''
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



# Import inference module
from inference import basic_inference
from train import train
from generator_360 import Generator360
from discriminator_360 import Discriminator360
from dataset import SitzmannDataset
import utils

if __name__ == "__main__":

	# Parse arguments
	parser = configargparse.ArgumentParser()
	parser.add_argument("--mode", required=True, type=str, default='inference', help="[Current] Running mode: --mode inference")
	opt = parser.parse_args()

	# Currently, there is only one admited parameter
	if opt.mode == 'inference':

		# Model currently developed to work on GPU
		device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		print("* Working on " + str(device))
		assert str(device) == "cuda:0"

		# Load model_generator   
		generator = torch.load('models/model_generator_217.pth')

		# Basic inference
		image_path = "data/test.jpg"		# The path where your image is
		path_to_save = "test/"				# The output path to save your image
		basic_inference(image_path=image_path, generator=generator, device=device, path_to_save=path_to_save, n_generated=50)
		print("Done.")

	elif opt.mode == 'train':

		# Set seed for reproducibility
		rSeed = 13579		

		# Establish seed
		print("* Random seed: ", rSeed)
		random.seed(rSeed)
		torch.manual_seed(rSeed)

		# Check if we run on GPU or CPU
		device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		print("* Working on " + str(device))
		assert str(device) == "cuda:0"

		# Instantiate dataset
		sitzmann_dataset = SitzmannDataset()
		dataloader = DataLoader(sitzmann_dataset, batch_size=utils.batch_size, shuffle=True, num_workers=0)

		# Instantiate networks
		netG = Generator360().to(device)
		netD = Discriminator360().to(device)

		# Apply the weights_init function to randomly initialize all weights.
		try:
			netG.apply(utils.weights_init)
			netD.apply(utils.weights_init)
		except:
			pass

		# Print the model
		print('* Generator:')
		print(netG)
		print('* Discriminator:')
		print(netD)

		# Initialize BCELoss function
		criterion = nn.BCELoss()

		# Create latent vector and get an image to visualize
		# generator process
		fixed_noise = torch.randn(utils.batch_size, utils.random_z, 1, 1, device=device)
		fixed_noise = fixed_noise.squeeze()

		# Establish convention for real and fake labels during training
		real_label = 1.
		fake_label = 0.

		# Setup Adam optimizers for both G and D
		optimizerD = optim.Adam(netD.parameters(), lr=utils.lr_d, betas=(0.5, 0.99))
		optimizerG = optim.Adam(netG.parameters(), lr=utils.lr_g, betas=(0.5, 0.99))

		# Train both models
		train(netG, netD, dataloader, criterion, real_label, fake_label, optimizerD, optimizerG, device)

