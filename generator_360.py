# Import Torch library
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as torch_models

# Import utils
import utils

# Import spherical convs
from CoordConv import AddCoordsTh
from spherenet import SphereConv2D

class Generator360(nn.Module):
    def __init__(self):
        super(Generator360, self).__init__()
   
        self.coord_conv = AddCoordsTh(x_dim=128,y_dim=256,with_r=False)

        # Image pipeline
        self.image_conv1 = SphereConv2D(5, 64, stride=2, bias=False)
        self.image_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.image_conv2 = SphereConv2D(64, 128, stride=2, bias=False)
        self.image_norm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv3 = SphereConv2D(128, 256, stride=2, bias=False)
        self.image_norm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv3_5 = SphereConv2D(256, 512, stride=2, bias=False)
        self.image_norm3_5 = nn.BatchNorm2d(512)
        self.leaky_relu3_5 = nn.LeakyReLU(0.2, inplace=True)

        # Joint pipeline
        self.sum_chan = 512 + 2

        self.image_conv4 = nn.Conv2d(self.sum_chan, 256, 4, 2, 1, bias=False)
        self.image_norm4 = nn.BatchNorm2d(256)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.image_conv5 = nn.Conv2d(256, 64, 4, 2, 1, bias=False)
        self.image_norm5 = nn.BatchNorm2d(64)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.fc1 = nn.Linear(64 * 4 * 2, 90)
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()

        self.activation = nn.Tanh()



    def forward(self, image, z, batch_size=utils.batch_size, debug=False):

        x = image
        y = z

        x = self.coord_conv(x)

        x = self.leaky_relu1(self.image_norm1(self.image_conv1(x)))

        x = self.leaky_relu2(self.image_norm2(self.image_conv2(x)))

        x = self.leaky_relu3(self.image_norm3(self.image_conv3(x)))

        x = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x)))
    
        y = torch.reshape(y, (batch_size, 2, 8, 16))

        # Joint operations
        x = torch.cat((y, x), dim=1)

        x = self.leaky_relu4(self.image_norm4(self.image_conv4(x)))

        x = self.leaky_relu5(self.image_norm5(self.image_conv5(x))) 

        x = self.activation(self.fc1(self.flatten(x)))

        return x

    