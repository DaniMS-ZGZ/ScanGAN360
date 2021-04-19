from __future__ import print_function
#%matplotlib inline
import argparse
import os
from pathlib import Path
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
import scipy.io
from scipy.interpolate import make_interp_spline, interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')
from scipy import ndimage
import scipy.stats as st

import utils
import copy
import cv2
import math
import operator


'''
Basic inference function.
Parameters:
	image_path = Directory path for the image to be tested.
	generator = Generator's model
	device = Device to run the inference on (only supports GPU at the moment)
	path_to_save = Directory path where results will be saved
	n_generated = Number of scanpaths to generate
'''
def basic_inference(image_path, generator, device, path_to_save, n_generated=50):

	# Set generator to eval mode
	generator.eval()
	# Create the path if does not exist
	Path(path_to_save).mkdir(parents=True, exist_ok=True)

	# Load image
	image = cv2.imread(image_path,cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Resize once to adjust to gaze point ratio
	original_image = image.copy()
	original_h, original_w, _ = image.shape
	ratio_w = original_w / utils.image_size[1]
	ratio_h = original_h / utils.image_size[0]

	# Resize image
	image = cv2.resize(image, (utils.image_size[1], utils.image_size[0]), interpolation=cv2.INTER_AREA)
	image = image.astype(np.float32) / 255.0

	# Normalize image
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
	b_image = transform(image)

	# Prefix to save your file with
	preffix = "inference_"

	# Number of batch (in this case, there is only one)
	n_batch = 0

	print("* [Batch %d] Data ready"%(n_batch))


	# Predict scanpaths and compute saliency
	_generated_scanpaths = []
	for n in range(n_generated):
		# Generate random noise from -1 to 1.
		noise = 2 * torch.randn(1, utils.random_z, 1, 1, device=device).squeeze() - 1
		with torch.no_grad():
			fake_latlon = generator(transform(image)[None, :, :, :].to(device), noise, 1, debug=False)
			fake_latlon = fake_latlon.detach().cpu().squeeze()
			# We are saving both the results in the range of 0-1 and 0-image_size
			_fake = []
			_n = []
			for i in range(0,len(fake_latlon),3):
				lat = np.arctan2(fake_latlon[i+2], np.sqrt(fake_latlon[i]**2 + fake_latlon[i+1]**2))
				lon = np.arctan2(fake_latlon[i+1], fake_latlon[i])
				# From lat-lon to x,y
				y = ((lat / (np.pi / 2) + 1)) / 2
				x = ((lon / np.pi) + 1) / 2
				# Save results in image space
				_fake.append(x * utils.image_size[1])
				_fake.append(y * utils.image_size[0])
			_generated_scanpaths.append(_fake)
		if n % 10 == 0:
			print("Generated %d scanpaths"%(n))


	print("* [Batch %d] generated %d scanpaths."%(n_batch, n_generated))



	# Change range to print more or less results... Each rep will print 25 scanpaths.
	for reps in range(0,2):
		# Plot predicted scanpaths
		fig, axs = plt.subplots(5,5)
		fig.set_size_inches(72,38)
		plt.title("Generated scanpaths")
		# Print GT examples
		for k in range(0, 25):
			idx1 = int(k / 5)
			idx2 = k % 5

			# Separate coordinates
			# Plot the last generated scanpath
			points_x = []
			points_y = []
			for i in range(0, len(_generated_scanpaths[25 * reps + k]), 2):
				points_x.append(_generated_scanpaths[25 * reps + k][i])
				points_y.append(_generated_scanpaths[25 * reps + k][i+1])

			colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

			# We save previous points to "correctly" draw the lines that join two consecutive points,
			# even if it jumps from one side to the other
			previous_point = None
			for num, x, y, c in zip(range(0,len(points_x)), points_x, points_y, colors):
				x = x * ratio_w
				y = y * ratio_h
				color = 'b'
				markersize = 28.
				if previous_point is not None:
					if abs(previous_point[0] - x) < (utils.image_size[1] * ratio_w)/2:
						axs[idx1, idx2].plot([x,previous_point[0]],[y,previous_point[1]],color='blue',linewidth=7.,alpha=0.35)
					else:
						# Join from the borders of the image (e.g., one point at the very left and the next at the very right)
						h_diff = (y - previous_point[1]) / 2
						if (x > previous_point[0]):			# X is on the right, Previous is on the Left
							axs[idx1, idx2].plot([previous_point[0],0],[previous_point[1],previous_point[1] + h_diff],color='blue',linewidth=7.,alpha=0.35)
							axs[idx1, idx2].plot([utils.image_size[1] * ratio_w,x], [previous_point[1] + h_diff,y],color='blue',linewidth=7.,alpha=0.35)
						else:
							axs[idx1, idx2].plot([previous_point[0],utils.image_size[1] * ratio_w],[previous_point[1],previous_point[1] + h_diff],color='blue',linewidth=7.,alpha=0.35)
							axs[idx1, idx2].plot([0,x], [previous_point[1] + h_diff,y],color='blue',linewidth=7.,alpha=0.35)
				previous_point = [x,y]
				axs[idx1, idx2].plot(x, y, marker='o', markersize=markersize, color=c, alpha=.8)
			axs[idx1, idx2].imshow(original_image)
			axs[idx1, idx2].axis('off')
			# axs[idx1, idx2].title.set_text("fDTW: %0.2f - REC: %0.2f"%(total_fdtw, total_rec))
		# plt.show()
		plt.savefig(path_to_save + preffix + "prediction_examples_" + str(reps) + ".png")
		plt.axis('off')
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.clf()
		plt.close('all')
		print("Set of images %d already printed."%(reps))

	print("* [Batch %d] generated scanpaths printed."%(n_batch))