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


# Import inference module
from inference import basic_inference


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

