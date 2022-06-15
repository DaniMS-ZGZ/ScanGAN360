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


class Discriminator360(nn.Module):
    def __init__(self):
        super(Discriminator360, self).__init__()


        '''
        self.lspace_y = np.linspace(0,1,utils.image_size[0])
        self.coord_conv_y = torch.from_numpy(np.array([self.lspace_y]*utils.image_size[1]).transpose()).unsqueeze(0).float().cuda()
        self.lspace_x = np.linspace(0,1,utils.image_size[1])
        self.coord_conv_x = torch.from_numpy(np.array([self.lspace_x]*utils.image_size[0])).unsqueeze(0).float().cuda()
        '''

        self.coord_conv = AddCoordsTh(x_dim=128,y_dim=256,with_r=False)

         # Image pipeline
        self.image_conv1 = SphereConv2D(5, 64, stride=2, bias=False)
        self.image_norm1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
            # Expected output: 64x64x64
        
        self.image_conv2 = SphereConv2D(64, 128, stride=2, bias=False)
        self.image_norm2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
            # Expected output: 32x32x128

        self.image_conv3 = SphereConv2D(128, 256, stride=2, bias=False)
        self.image_norm3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
            # Expected output: 16x16x512

        self.image_conv3_5 = SphereConv2D(256, 512, stride=2, bias=False)
        self.image_norm3_5 = nn.BatchNorm2d(512)
        self.leaky_relu3_5 = nn.LeakyReLU(0.2, inplace=True)
            # Expected output: 16x16x512

        self.sum_chan = 512 + 90

        self.image_conv4 = nn.Conv2d(self.sum_chan, 256, 4, 2, 1, bias=False)
        self.image_norm4 = nn.BatchNorm2d(256)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
            # Expected output: 16x16x512

        self.image_conv5 = nn.Conv2d(256, 64, 4, 2, 1, bias=False)
        self.image_norm5 = nn.BatchNorm2d(64)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)
            # Expected output: 8x8x512

        self.fc1 = nn.Linear(64 * 4 * 2, 1)
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()


            

    def forward(self, image, scanpath, debug=False):

        # x = image.permute(0, 3, 1, 2)
        x = image
        y = scanpath      # Get first 


        
        # Image pipeline
        if debug:
            print("[D][00] X Shape: %s"%str(x.shape))
            print("[D][01] S Shape: %s"%str(y.shape))

        # Concat CoordConv
        '''
        x = torch.cat((x, self.coord_conv_x.repeat(image.shape[0],1,1,1)), dim=1)
        x = torch.cat((x, self.coord_conv_y.repeat(image.shape[0],1,1,1)), dim=1)
        '''
        x = self.coord_conv(x)

        x = self.leaky_relu1(self.image_norm1(self.image_conv1(x)))
        if debug:
            print("[D][02] X Shape: %s"%str(x.shape))

        x = self.leaky_relu2(self.image_norm2(self.image_conv2(x)))
        if debug:
            print("[D][03] X Shape: %s"%str(x.shape))

        x = self.leaky_relu3(self.image_norm3(self.image_conv3(x)))
        if debug:
            print("[D][03] X Shape: %s"%str(x.shape))

        x = self.leaky_relu3_5(self.image_norm3_5(self.image_conv3_5(x)))
        if debug:
            print("[G][04] X Shape: %s"%str(x.shape))
        
        '''
        x = self.vgg19.features(x)
        if debug:
            print("[G][02] X Shape: %s"%str(x.shape))
        '''


        # Join scanpath and convs
        _y = y.repeat(8, 16, 1, 1).permute(2, 3, 0, 1)
        # Y shape is batch x npoitns x 16 x 16

        x = torch.cat((x, _y),dim=1)
        if debug:
            print("[D][INT] X Shape: %s"%str(x.shape))

        x = self.leaky_relu4(self.image_norm4(self.image_conv4(x)))
        if debug:
            print("[D][04] X Shape: %s"%str(x.shape))

        x = self.leaky_relu5(self.image_norm5(self.image_conv5(x)))
        if debug:
            print("[D][05] X Shape: %s"%str(x.shape))
        
        x = self.fc1(self.flatten(x))
        if debug:
            print("[D][07] X Shape: %s"%str(x.shape))
        

        return self.activation(x)

        '''

        # Noise pipeline
        # y = self.leaky_relu3(self.fc1(y))
        # if debug:
        #    print("[D][04] S Shape: %s"%str(y.shape))

        y = y.repeat(8,8,1,1).permute(2, 3, 0, 1)
        if debug:
            print("[D][05] S Shape: %s"%str(y.shape))

        y = self.noise_relu1(self.noise_norm1(self.noise_conv1(y)))
        if debug:
            print("[D][06] S Shape: %s"%str(y.shape))

        y = self.noise_relu2(self.noise_norm2(self.noise_conv2(y)))
        if debug:
            print("[D][07] S Shape: %s"%str(y.shape))

        # Joint operations
        x = torch.cat((x, y), dim=1)
        if debug:
            print("[D][08] X Shape: %s"%str(x.shape))

        x = self.join_relu1(self.join_norm1(self.join_conv1(x)))
        if debug:
            print("[D][09] X Shape: %s"%str(x.shape))

        x = self.join_relu2(self.join_norm2(self.join_conv2(x)))
        if debug:
            print("[D][10] X Shape: %s"%str(x.shape))

        x = self.join_relu3(self.join_norm3(self.join_conv3(x)))
        if debug:
            print("[D][11] X Shape: %s"%str(x.shape))

        x = self.activation(self.adaptive_avg_pool(x))
        if debug:
            print("[D][12] X Shape: %s"%str(x.shape))


        return x


        '''

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.ConvTranspose2d) or isinstance(m_to, nn.Conv2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

