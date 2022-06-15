import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from scipy import ndimage

import os, sys
import utils
import collections

import cv2




# Dataset based on Sitzmann et al's
class SitzmannDataset():
	def __init__(self, images_path=None, gaze_path=None):
		# Path to images 
		self.images_path = utils.images_path
		# Path to gaze data
		self.gaze_path = utils.gaze_path
		# List with all image paths
		self.images = []
		# TUples with image and scanpath
		self.image_and_scanpath = []
		# Dictionary with all scanpaths from an image
		self.image_and_scanpath_dict = {}
		# Transforms
		self.transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

		
		for file_name in os.listdir(self.images_path):
			if ".png" in file_name:
				self.images.append(os.path.join(self.images_path, file_name))

		print("* Loaded %d images from Siztmann DATASET"%len(self.images))

		for im in self.images:
			self.image_and_scanpath_dict[im] = []

		# Now iterate over all gaze data and associate them to its corresponding scene
		for file_name in os.listdir(self.gaze_path):
			if ".csv" in file_name and not "head" in file_name:
				image_name = file_name[:-4]
				if not os.path.join(self.images_path, image_name) + ".png" in self.images:
					continue
				f = open(os.path.join(self.gaze_path,file_name), "r")
				i = 0
				for row in f:
					if (i < 10):
						assert str(i) == row[0]
						row = row[2:]
					else:
						assert str(i) == row[0:2]
						row = row[3:]
					# Each row is a different user
					points = row.replace(" ","").replace("\n","").split(",")
					points = [float(p) for p in points]

					# Instead of taking all the >700 points, get only 70 of them.
					points = points[0:1400]
					_points = []
					for j in range(0,len(points),48):		#48 for 30
						_points.append(points[j])
						_points.append(points[j+1])
					
					# _points = _points[:120]

					path = self.images_path + image_name + ".png"
					self.image_and_scanpath.append((path, _points))
					self.image_and_scanpath_dict[path].append(_points)

					# Counter
					i = i + 1	

		print("* Loaded %d tuples of image and scanpaths from Siztmann dataset"%len(self.image_and_scanpath))

		# Split dataset in smaller mini-batches to fit GPU
		# E.G. each mini-batch will have one image and 10 scapaths
		self.mini_batch_images = []
		for e in self.images:
			total_scanpaths = len(self.image_and_scanpath_dict[e])
			chunks = [self.image_and_scanpath_dict[e][x:x+8] for x in range(0, total_scanpaths, 8)] 
			for c in chunks:
				self.mini_batch_images.append([e,c])

		print("* Dataset containing %d images, %d tuples I-S, and %d mini-batches of 8. \nWe are using then %d mini-batches in a whole batch."%(len(self.images), len(self.image_and_scanpath), len(self.mini_batch_images), utils.real_batch_size))


	def __len__(self):
		return len(self.mini_batch_images)

		# Each batch will contain all scanpaths from an image
	def __getitem__(self, idx):
		image_path = self.mini_batch_images[idx][0]
		scanpaths = self.mini_batch_images[idx][1]

		# Load i-th image
		image = cv2.imread(image_path,cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# Resize once to adjust to gaze point ratio
		image = cv2.resize(image, (4096, 2048), interpolation=cv2.INTER_AREA)
		original_h, original_w, _ = image.shape
		ratio_w = original_w / utils.image_size[1]
		ratio_h = original_h / utils.image_size[0]
		image = cv2.resize(image, (utils.image_size[1], utils.image_size[0]), interpolation=cv2.INTER_AREA)
		image = image.astype(np.float32) / 255.0
		o_image = image
		# print(image.shape)


		res = [[], []]
		for n_s in range(len(scanpaths)):
			# Modify each scanpath
			scanpath = scanpaths[n_s]

			# Load i-th scanpath
			flat_scanpath = []
			for i in range(0, len(scanpath), 2):
				x = scanpath[i+1]
				y = scanpath[i]
				flat_scanpath.append(x / ratio_w)
				flat_scanpath.append(y / ratio_h)
			
			'''
			# Separate coordinates
			points_x = []
			points_y = []
			for i in range(0,len(flat_scanpath),2):
				points_x.append(flat_scanpath[i])
				points_y.append(flat_scanpath[i+1])

			colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

			for x, y, c in zip(points_x, points_y, colors):
				plt.plot(x, y,  marker='o', color=c)
			# plt.plot(true_points, marker='o', color='r')
			plt.imshow(image)
			plt.show()
			'''
		
			# Normalize data (0 being 0px, 1 being utils.image_size[0] or utils.image_size[1])
			# print(len(flat_scanpath))
			# print(flat_scanpath)
			for i in range(0,len(flat_scanpath),2):
				# points_x.append(_flat_scanpath[i])
				# points_y.append(_flat_scanpath[i+1])
				flat_scanpath[i] = flat_scanpath[i] / utils.image_size[1]
				flat_scanpath[i] = ((flat_scanpath[i] * 2) - 1) * (np.pi - 1e-2)
				flat_scanpath[i+1] = flat_scanpath[i+1] / utils.image_size[0]
				flat_scanpath[i+1] =  ((flat_scanpath[i+1] * 2) - 1) * (np.pi/2 - 1e-2)

			# print(flat_scanpath)

			# Map them to a three coordinate system
			three_coord = []
			for i in range(0,len(flat_scanpath),2):
				lon = flat_scanpath[i]
				lat = flat_scanpath[i+1]
				x = np.cos(lat) * np.cos(lon)
				y = np.cos(lat) * np.sin(lon)
				z = np.sin(lat) 
				three_coord.append(x)
				three_coord.append(y)
				three_coord.append(z)

			'''
			# Separate coordinates
			points_x = []
			points_y = []
			for i in range(0,len(three_coord),3):
				lat = np.arctan2(three_coord[i+2], np.sqrt(three_coord[i]**2 + three_coord[i+1]**2))
				lon = np.arctan2(three_coord[i+1], three_coord[i])
				# From lat-lon to x,y
				y = ((lat / (np.pi / 2) + 1)) / 2
				x = ((lon / np.pi) + 1) / 2
				# Show
				points_x.append(x * utils.image_size[1])
				points_y.append(y * utils.image_size[0])

			colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

			for x, y, c in zip(points_x, points_y, colors):
				plt.plot(x, y,  marker='o', color=c)
			# plt.plot(true_points, marker='o', color='r')
			plt.imshow(image)
			plt.show()
			'''

			res[0].append(self.transform(image))
			res[1].append(torch.FloatTensor(three_coord))
			res.append(o_image)

		# Return both information
		return res



