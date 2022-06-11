import enum
import math
import os
import sys
import math
import io
import json
import platform
from enum import Enum
from copy import deepcopy
from click import style
import torch
from torch._C import parse_ir
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from mlflow import log_metric, log_param, log_artifacts

sys.path.insert(0, os.path.abspath(''))
sys.path.insert(0, os.path.abspath('..'))
from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDictJson, WriteDictJson, str2bool
from pymlutil.s3 import s3store, Connect
from pymlutil.functions import GaussianBasis
from networks.totalloss import TotalLoss

# Inner neural architecture cell repetition structure
# Process: Con2d, optional batch norm, optional ReLu


class ConvBR(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 batch_norm=True, 
                 relu=True,
                 kernel_size=3, 
                 stride=1,
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros',
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 dropout_rate=0.2,
                 sigmoid_scale = 5.0, # Channel sigmoid scale factor
                 search_structure=True,
                 residual = False,
                 dropout=False,
                 conv_transpose=False, # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
                 k_prune_sigma=0.33,
                 device=torch.device("cpu"),
                 ):
        super(ConvBR, self).__init__()
        self.in_channels = in_channels
        if out_channels < 1:
            raise ValueError("out_channels must be > 0")
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.relu = relu
        self.weight_gain = abs(weight_gain)
        self.convMaskThreshold = convMaskThreshold
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.dropout_rate = dropout_rate
        self.sigmoid_scale = sigmoid_scale
        self.search_structure = search_structure
        self.residual = residual
        self.use_dropout = dropout
        self.conv_transpose = conv_transpose
        self.k_prune_sigma=k_prune_sigma
        self.device = device

        self.channel_scale = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float, device=self.device))

        if type(kernel_size) == int:
            padding = kernel_size // 2 # dynamic add padding based on the kernel_size
        else:
            padding = kernel_size//2

        if not self.conv_transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        else:
            self.conv = nn.ConvTranspose2d( in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size, 
                                            stride=stride, 
                                            groups=groups, 
                                            bias=bias, 
                                            dilation=dilation, 
                                            padding_mode=padding_mode)

        if self.batch_norm:
            self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.sigmoid = nn.Sigmoid()
        self.total_trainable_weights = model_weights(self)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

        if self.relu:
            self.activation = nn.ReLU()
        else:
            self.activation = None

            

        self._initialize_weights()

    def _initialize_weights(self):
        #nn.init.normal_(self.channel_scale, mean=0.5,std=0.33)
        nn.init.ones_(self.channel_scale)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.batch_norm and self.batchnorm2d and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def ApplyParameters(self, search_structure=None, convMaskThreshold=None, dropout=None,
                        sigmoid_scale=None, weight_gain=None):
        if search_structure is not None:
            if self.search_structure == False and search_structure == True:
                #nn.init.normal_(self.channel_scale, mean=0.5,std=0.33)
                nn.init.ones_(self.channel_scale)
            self.search_structure = search_structure
        if convMaskThreshold is not None:
            self.convMaskThreshold = convMaskThreshold
        if dropout is not None:
            self.use_dropout = dropout
        if sigmoid_scale is not None:
            self.sigmoid_scale = sigmoid_scale
        if weight_gain is not None:
            self.weight_gain = weight_gain

    def forward(self, x):
        if self.out_channels > 0:
            if self.use_dropout:
                x = self.dropout(x)
                
            x = self.conv(x)
  
            if self.batch_norm:
                x = self.batchnorm2d(x)

            if self.search_structure: #scale channels based on self.channel_scale
                    weight_scale = self.sigmoid(self.sigmoid_scale*self.channel_scale)[None,:,None,None]
                    x *= weight_scale  

            if self.activation:
                x = self.activation(x)
        else :
            print("Failed to prune zero size convolution")

        return x

    def ArchitectureWeights(self):
        if self.search_structure:
            weight_basis = GaussianBasis(self.channel_scale)
            weight_scale = self.sigmoid(self.sigmoid_scale*self.channel_scale)

            '''if not self.batch_norm:
                # Include convolution norm in channel pruning weight if no batch norm
                if self.conv_transpose:
                    den = list(self.conv.weight.shape)
                    del den[1]
                    den = np.sqrt(np.prod(den))
                    norm = torch.linalg.norm(self.conv.weight , dim=(0,2,3))/den
                else:
                    den = list(self.conv.weight.shape)
                    del den[0]
                    den = np.sqrt(np.prod(den))
                    norm = torch.linalg.norm(self.conv.weight , dim=(1,2,3))/den
                conv_scale = torch.tanh(self.weight_gain*norm)
                conv_weights = weight_scale*conv_scale
            else:
                conv_weights = weight_scale '''
            conv_weights = weight_scale

        else:
            weight_basis = torch.zeros_like(self.channel_scale, device=self.device)
            conv_weights = torch.ones_like(self.channel_scale, device=self.device)


        cell_weights = model_weights(self)

        if self.out_channels > 0:
            # Keep sum as [1] tensor so subsequent concatenation works
            architecture_weights = (cell_weights/ self.out_channels) * conv_weights.sum_to_size((1))
        else:
            architecture_weights = conv_weights.sum_to_size((1))

        return architecture_weights, cell_weights, conv_weights, weight_basis

    # Remove specific network dimensions
    # remove dimension where inDimensions and outDimensions arrays are 0 for channels to be removed
    def ApplyStructure(self, in_channel_mask=None, out_channel_mask=None, msg=None):
        if msg is None:
            prefix = "ConvBR::ApplyStructure"
        else:
            prefix = "ConvBR::ApplyStructure {}".format(msg)

        # Adjust in_channel size based on in_channel_mask
        if in_channel_mask is not None:
            if len(in_channel_mask) == self.in_channels:
                if self.conv_transpose:
                    self.conv.weight.data = self.conv.weight[in_channel_mask!=0, :]
                else:
                    self.conv.weight.data = self.conv.weight[:, in_channel_mask!=0]
                self.in_channels = len(in_channel_mask[in_channel_mask!=0])
            else:
                raise ValueError("{} len(in_channel_mask)={} must be equal to self.in_channels={}".format(prefix, len(in_channel_mask), self.in_channels))

        # Convolution norm gain mask
        if self.in_channels ==0: # No output if no input
            conv_mask = torch.zeros((self.out_channels), dtype=torch.bool, device=self.device)
        elif self.search_structure:
            conv_mask = self.sigmoid(self.sigmoid_scale*self.channel_scale)

            '''if self.batch_norm:
                conv_mask = self.sigmoid(self.sigmoid_scale*self.channel_scale)
            else:
                if self.conv_transpose:
                    den = list(self.conv.weight.shape)
                    del den[1]
                    den = np.sqrt(np.prod(den))
                    norm = torch.linalg.norm(self.conv.weight , dim=(0,2,3))/den
                else:
                    den = list(self.conv.weight.shape)
                    del den[0]
                    den = np.sqrt(np.prod(den))
                    norm = torch.linalg.norm(self.conv.weight , dim=(1,2,3))/den

                norm = torch.linalg.norm(self.conv.weight, dim=(1,2,3))/np.sqrt(np.product(self.conv.weight.shape[1:]))
                conv_mask *= torch.tanh(self.weight_gain*norm)'''

            conv_mask = conv_mask > self.convMaskThreshold
        else:
            conv_mask = self.channel_scale > float('-inf') # Always true if self.search_structure == False

        if out_channel_mask is not None:
            if len(out_channel_mask) == self.out_channels:
                # If either mask is false, then the convolution is removed (not nand)
                conv_mask = torch.logical_not(torch.logical_or(torch.logical_not(conv_mask), torch.logical_not(out_channel_mask)))
                
            else:
                raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))

        prev_convolutions = len(conv_mask)
        pruned_convolutions = len(conv_mask[conv_mask==False])

        if self.search_structure:
            if self.residual and prev_convolutions==pruned_convolutions: # Pruning full convolution 
                conv_mask = ~conv_mask # Residual connection becomes a straight through connection
            else:
                self.conv.bias.data = self.conv.bias[conv_mask!=0]
                if self.conv_transpose:
                    self.conv.weight.data = self.conv.weight[:, conv_mask!=0]
                else:
                    self.conv.weight.data = self.conv.weight[conv_mask!=0]

                self.channel_scale.data = self.channel_scale.data[conv_mask!=0]

                if self.batch_norm:
                    self.batchnorm2d.bias.data = self.batchnorm2d.bias.data[conv_mask!=0]
                    self.batchnorm2d.weight.data = self.batchnorm2d.weight.data[conv_mask!=0]
                    self.batchnorm2d.running_mean = self.batchnorm2d.running_mean[conv_mask!=0]
                    self.batchnorm2d.running_var = self.batchnorm2d.running_var[conv_mask!=0]

            self.out_channels = len(conv_mask[conv_mask!=0])

        print("{} {}={}/{} in_channels={} out_channels={}".format(prefix, pruned_convolutions/prev_convolutions, pruned_convolutions, prev_convolutions, self.in_channels, self.out_channels))

        return conv_mask


DefaultMaxDepth = 1
class Cell(nn.Module):
    def __init__(self,
                 in1_channels, 
                 in2_channels = 0,
                 batch_norm=False, 
                 relu=True,
                 kernel_size=3, 
                 padding=0, 
                 dilation=1, 
                 groups=1,
                 bias=True, 
                 padding_mode='zeros',
                 residual=True,
                 dropout=False,
                 device=torch.device("cpu"),
                 feature_threshold=0.5,
                 cell_convolution=DefaultMaxDepth,
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 search_structure=True, 
                 convolutions=[{'out_channels':64, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True, 'conv_transpose':False}],
                 dropout_rate = 0.2,
                 sigmoid_scale = 5.0,
                 k_prune_sigma=0.33
                 ):
                
        super(Cell, self).__init__()

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.batch_norm = batch_norm
        self.relu = relu
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.residual = residual
        self.dropout = dropout
        self.device = device
        self.feature_threshold = feature_threshold
        self.search_structure = search_structure
        self.cell_convolution = nn.Parameter(torch.tensor(cell_convolution, dtype=torch.float, device=self.device))
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold
        self.convolutions = deepcopy(convolutions)
        self.dropout_rate = dropout_rate
        self.sigmoid_scale=sigmoid_scale
        self.k_prune_sigma=k_prune_sigma


        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convolutions uses out_channels as chanels

        src_channels = in_chanels = self.in1_channels+self.in2_channels

        totalStride = 1
        totalDilation = 1

        for i, convdev in enumerate(convolutions):
            conv_transpose = False
            if 'conv_transpose' in convdev and convdev['conv_transpose']:
                conv_transpose = True

            conv = ConvBR(src_channels, convdev['out_channels'], 
                batch_norm=self.batch_norm, 
                relu=self.relu, 
                kernel_size=convdev['kernel_size'], 
                stride=convdev['stride'], 
                dilation=convdev['dilation'],
                groups=self.groups, 
                bias=self.bias, 
                padding_mode=self.padding_mode,
                weight_gain=self.weight_gain,
                convMaskThreshold=self.convMaskThreshold, 
                dropout_rate=self.dropout_rate,
                search_structure=convdev['search_structure'] and self.search_structure ,
                sigmoid_scale = self.sigmoid_scale,
                residual=False, 
                dropout=self.dropout,
                conv_transpose=conv_transpose,
                k_prune_sigma=self.k_prune_sigma,
                device=self.device)
            self.cnn.append(conv)

            src_channels = convdev['out_channels']
            totalStride *= convdev['stride']
            totalDilation *= convdev['dilation']
 
        if self.residual and (in_chanels != self.convolutions[-1]['out_channels'] or totalStride != 1 or totalDilation != 1):
            self.conv_residual = ConvBR(in_chanels, self.convolutions[-1]['out_channels'], 
                batch_norm=self.batch_norm, 
                relu=self.relu, 
                kernel_size=1, 
                stride=totalStride, 
                dilation=totalDilation, 
                groups=self.groups, 
                bias=self.bias, 
                padding_mode=self.padding_mode,
                weight_gain=self.weight_gain,
                convMaskThreshold=self.convMaskThreshold, 
                dropout_rate=self.dropout_rate,
                search_structure=False,
                residual=True,
                dropout=self.dropout,
                k_prune_sigma=self.k_prune_sigma,
                device=self.device)
        else:
            self.conv_residual = None


        self._initialize_weights()
        self.total_trainable_weights = model_weights(self)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def ApplyParameters(self, search_structure=None, convMaskThreshold=None, dropout=None,
                        weight_gain=None, sigmoid_scale=None, feature_threshold=None,
                        k_prune_sigma=None): # Apply a parameter change
        if search_structure is not None:
            self.search_structure = search_structure

        if convMaskThreshold is not None:
            self.convMaskThreshold = convMaskThreshold

        if dropout is not None:
            self.use_dropout = dropout

        if weight_gain is not None:
            self.weight_gain = weight_gain
        if sigmoid_scale is not None:
            self.sigmoid_scale = sigmoid_scale
        if feature_threshold is not None:
            self.feature_threshold = feature_threshold

        if k_prune_sigma is not None:
            self.k_prune_sigma = k_prune_sigma

        if self.cnn is not None and len(self.cnn) > 0:
            for conv in self.cnn:
                conv.ApplyParameters(search_structure=search_structure, convMaskThreshold=convMaskThreshold, dropout=dropout,
                                     weight_gain=weight_gain, sigmoid_scale=sigmoid_scale)

    def ArchitectureWeights(self):
        architecture_weights = []
        layer_weights = []
        conv_weights = []
        weight_basises = []
        norm_conv_weight = []
        search_structure = []
        unallocated_weights  = torch.zeros((1), device=self.device)
        if self.cnn is not None:
            for i, l in enumerate(self.cnn): 
                layer_weight, cnn_weight, conv_weight, weight_basis = l.ArchitectureWeights()
                conv_weights.append(conv_weight)
                weight_basises.append(weight_basis)

                if self.convolutions[i]['search_structure']:
                    architecture_weights.append(layer_weight)
                    norm_conv_weight.append(layer_weight/cnn_weight)
                else:
                    unallocated_weights += cnn_weight

            if len(architecture_weights) > 0:
                architecture_weights = torch.cat(architecture_weights)
                architecture_weights = architecture_weights.sum_to_size((1))
                num_conv_weights = len(norm_conv_weight)
                if num_conv_weights > 0:
                    # Allocate remaining weights if pruning full channel
                    prune_weight = torch.cat(norm_conv_weight).min()
                    prune_weight = torch.tanh(self.weight_gain*prune_weight)
                    architecture_weights += unallocated_weights*prune_weight
            else: # Nothing to prune here
                architecture_weights = unallocated_weights
                prune_weight = torch.tensor(1.0, device=self.device)
            prune_weight = torch.tensor(1.0, device=self.device) # Disable prune weight from ConvBR weights

        else:
            architecture_weights = torch.zeros((1), device=self.device)
            prune_weight = torch.tensor(1.0, device=self.device)

        cell_weights = {'prune_weight':prune_weight, 'cell_weight':conv_weights, 'weight_basis': weight_basises}

        return architecture_weights, model_weights(self), cell_weights

    def ApplyStructure(self, in1_channel_mask=None, in2_channel_mask=None, msg=None, prune=None):

        # Reduce channels
        in_channel_mask = None
        out_channels = 0
        if in1_channel_mask is not None or in2_channel_mask is not None:

            if in1_channel_mask is not None:
                if len(in1_channel_mask) == self.in1_channels:
                    self.in1_channels = len(in1_channel_mask[in1_channel_mask!=0])
                else:
                    raise ValueError("ApplyStructure len(in1_channel_mask)={} must be = self.in1_channels {}".format(len(in1_channel_mask), self.in1_channels))

                in_channel_mask = in1_channel_mask

            if in2_channel_mask is not None:
                if len(in2_channel_mask) == self.in2_channels:
                    self.in2_channels = len(in2_channel_mask[in2_channel_mask!=0])
                else:
                    raise ValueError("ApplyStructure len(in2_channel_mask)={} must be = self.in2_channels {}".format(len(in2_channel_mask), self.in2_channels))
                
                if in1_channel_mask is not None:
                    in_channel_mask = torch.cat((in1_channel_mask, in2_channel_mask))

        else: # Do not reduce input channels.  
            in_channel_mask = torch.ones((self.in1_channels+self.in2_channels), dtype=torch.bool, device=self.device)
        
        if self.cnn is not None:
            out_channel_mask = in_channel_mask

            _, _, conv_weight  = self.ArchitectureWeights()
            if (prune is not None and prune) or conv_weight['prune_weight'].item() < self.feature_threshold: # Prune convolutions prune_weight is < feature_threshold
                out_channels = self.cnn[-1].out_channels
                out_channel_mask = torch.zeros((out_channels), dtype=np.bool, device=self.device)
                self.cnn = None
                layermsg = "Prune cell because prune_weight {} < feature_threshold {}".format(conv_weight['prune_weight'], self.feature_threshold)
                if msg is not None:
                    layermsg = "{} {} ".format(msg, layermsg)
                print(layermsg)
            else:
                for i, cnn in enumerate(self.cnn):
                    layermsg = "convolution {}/{} search_structure={}".format(i, len(self.cnn), cnn.search_structure)

                    if msg is not None:
                        layermsg = "{} {}".format(msg, layermsg)
                        
                    out_channel_mask = cnn.ApplyStructure(in_channel_mask=out_channel_mask, msg=layermsg)
                    out_channels = cnn.out_channels
                    if out_channels == 0: # Prune convolutions if any convolution has no more outputs
                        if i != len(self.cnn)-1: # Make mask the size of the cell output with all values 0
                            out_channel_mask = torch.zeros(self.cnn[-1].out_channels, dtype=np.bool, device=self.device)
     
                        self.cnn = None
                        out_channels = 0

                        layermsg = "Prune cell because convolution {} out_channels == {}".format(i, out_channels)
                        if msg is not None:
                            layermsg = "{} {} ".format(msg, layermsg)
                        print(layermsg)
                        break
        else:
            out_channel_mask = None
            out_channels = 0

        if self.conv_residual is not None:
            layermsg = "cell residual search_structure={}".format(self.conv_residual.search_structure)
            if msg is not None:
                layermsg = "{} {} ".format(msg, layermsg)
            
            out_channel_mask = self.conv_residual.ApplyStructure(in_channel_mask=in_channel_mask, out_channel_mask=out_channel_mask, msg=layermsg)
            out_channels = self.conv_residual.out_channels


        layermsg = "cell summary: weights={} in1_channels={} in2_channels={} out_channels={} residual={} search_structure={}".format(
            self.total_trainable_weights, 
            self.in1_channels, 
            self.in2_channels, 
            out_channels,
            self.residual,
            self.search_structure)
        if msg is not None:
            layermsg = "{} {} ".format(msg, layermsg)
        print(layermsg)

        return out_channel_mask


    def forward(self, in1, in2 = None, isTraining=False):
        if in1 is not None and in2 is not None:
            u = torch.cat((in1, in2), dim=1)
        elif in1 is not None:
            u = in1
        elif in2 is not None:
            u = in2
        else:
            return None

        # Resizing convolution
        if self.residual:
            if self.conv_residual and u is not None:
                residual = self.conv_residual(u)
            else:
                residual = u
        else:
            residual = None

        if self.cnn is not None:
            x = u
            for i, l in enumerate(self.cnn):
                x = self.cnn[i](x)

            if self.residual:
                y = x + residual
            else:
                y = x
        else:
            y = residual

        return y

class FC(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,

                 ):
        super(FC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.fc = nn.Linear(in_channels, self.out_channels)
        self.total_trainable_weights = model_weights(self)

    def forward(self, x):
        y = self.fc(x)
        return y

    def ArchitectureWeights(self):
        architecture_weights = model_weights(self)

        return architecture_weights, self.total_trainable_weights

    # Remove specific network dimensions
    # remove dimension where inDimensions and outDimensions arrays are 0 for channels to be removed
    def ApplyStructure(self, in_channel_mask=None, out_channel_mask=None, msg=None):
        if in_channel_mask is not None:
            if len(in_channel_mask) != self.in_channels:
                raise ValueError("len(in_channel_mask)={} must be equal to self.in_channels={}".format(len(in_channel_mask), self.in_channels))
        else: # Do not reduce input channels.  
            in_channel_mask = torch.ones(self.in_channels, dtype=torch.int32, device=self.device)

        if out_channel_mask is not None:
            if len(out_channel_mask) != self.out_channels:
                  raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))
        else: # Do not reduce input channels.  
            out_channel_mask = torch.ones(self.out_channels, dtype=torch.int32, device=self.device)

        pruned_convolutions = len(out_channel_mask[out_channel_mask==False])
        if pruned_convolutions > 0:
            numconvolutions = len(out_channel_mask)
            print("Pruned {}={}/{} convolutional weights".format(pruned_convolutions/numconvolutions, pruned_convolutions, numconvolutions))

        self.fc.weight.data = self.fc.weight.data[:,in_channel_mask!=0]

        self.fc.bias.data = self.fc.bias.data[out_channel_mask!=0]
        self.fc.weight.data = self.fc.weight.data[out_channel_mask!=0]

        prev_in_convolutions = len(in_channel_mask)
        self.in_channels = len(in_channel_mask[in_channel_mask!=0])

        prev_out_convolutions = len(out_channel_mask)
        self.out_channels = len(out_channel_mask[out_channel_mask!=0])

        if msg is None:
            prefix = "FC::ApplyStructure"
        else:
            prefix = "FC::ApplyStructure {}".format(msg)
        print("{} in {}={}/{} out {}={}/{} convolutions in_channels={} out_channels={}".format(
            prefix, 
            (prev_in_convolutions-self.in_channels)/prev_in_convolutions, 
            (prev_in_convolutions-self.in_channels),
            prev_in_convolutions,
            (prev_out_convolutions-self.out_channels)/prev_out_convolutions, 
            (prev_out_convolutions-self.out_channels),
            prev_out_convolutions,
            self.in_channels, 
            self.out_channels))



        return out_channel_mask

# Resenet definitions from https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
class Resnet(Enum):
    layers_18 = 18
    layers_34 = 34
    layers_50 = 50
    layers_101 = 101
    layers_152 = 152
    cifar_20 = 20
    cifar_32 = 32
    cifar_44 = 44
    cifar_56 = 56
    cifar_110 = 110


def ResnetCells(size = Resnet.layers_50):
    resnetCells = []
    
    sizes = {
        'layers_18': [2, 2, 2, 2], 
        'layers_34': [3, 4, 6, 3], 
        'layers_50': [3, 4, 6, 3], 
        'layers_101': [3, 4, 23, 3], 
        'layers_152': [3, 8, 36, 3],
        'cifar_20' : [7,6,6],
        'cifar_32' : [11,10,10],
        'cifar_44' : [15,14,14],
        'cifar_56' : [19,18,18],
        'cifar_110' : [37,36,36],
        }
    bottlenecks = {
        'layers_18': False, 
        'layers_34': False, 
        'layers_50': True, 
        'layers_101': True, 
        'layers_152': True,
        'cifar_20': False, 
        'cifar_32': False, 
        'cifar_44': False,
        'cifar_56': False, 
        'cifar_110': False
        }

    cifar_style = {
        'layers_18': False, 
        'layers_34': False, 
        'layers_50': False, 
        'layers_101': False, 
        'layers_152': False,
        'cifar_20': True, 
        'cifar_32': True, 
        'cifar_44': True,
        'cifar_56': True, 
        'cifar_110': True
        }

    resnet_cells = []
    block_sizes = sizes[size.name]
    bottleneck = bottlenecks[size.name]
    is_cifar_style = cifar_style[size.name]

    cell = []
    if is_cifar_style:
        network_channels = [32, 16, 8]
        cell.append({'out_channels':network_channels[0], 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':False})
    else:
        network_channels = [64, 128, 256, 512]
        cell.append({'out_channels':network_channels[0], 'kernel_size': 7, 'stride': 3, 'dilation': 1, 'search_structure':False})

    resnetCells.append({'residual':False, 'cell':cell})

    for i, layer_size in enumerate(block_sizes):
        block_channels = network_channels[i]
        for j in range(layer_size):
            stride = 1
            # Downsample by setting stride to 2 on the first layer of each block
            if i != 0 and j == 0:
                stride = 2
            cell = []
            if bottleneck:
                # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
                # while original implementation places the stride at the first 1x1 convolution(self.conv1)
                # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
                # This variant is also known as ResNet V1.5 and improves accuracy according to
                # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
                cell.append({'out_channels':network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':True})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1, 'search_structure':True})
                cell.append({'out_channels':4*network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':False})
            else:
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1, 'search_structure':True})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':False})
            resnetCells.append({'residual':True, 'cell':cell})
        
    return resnetCells


class Classify(nn.Module):
    def __init__(self, convolutions, 
    device=torch.device("cpu"), source_channels = 3, out_channels = 10, initial_channels=16, 
    batch_norm=True, weight_gain=11, convMaskThreshold=0.5, 
    dropout_rate=0.2, search_structure = True, sigmoid_scale=5.0, feature_threshold=0.5):
        super().__init__()
        self.device = device
        self.source_channels = source_channels
        self.out_channels = out_channels
        self.initial_channels = initial_channels
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.search_structure = search_structure
        self.sigmoid_scale = sigmoid_scale
        self.feature_threshold = feature_threshold
                
        self.cells = torch.nn.ModuleList()
        in_channels = self.source_channels

        for i, cell_convolutions in enumerate(convolutions):

            convdfn = None

            cell = Cell(in1_channels=in_channels, 
                batch_norm=self.batch_norm,
                device=self.device,  
                weight_gain = self.weight_gain,
                convMaskThreshold = self.convMaskThreshold,
                residual=cell_convolutions['residual'],
                convolutions=cell_convolutions['cell'],  
                dropout_rate=self.dropout_rate, 
                search_structure=self.search_structure, 
                sigmoid_scale=self.sigmoid_scale, 
                feature_threshold=self.feature_threshold)
            in_channels = cell_convolutions['cell'][-1]['out_channels']
            self.cells.append(cell)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FC(in_channels, self.out_channels)

        self.total_trainable_weights = model_weights(self)

        self.fc_weights = model_weights(self.fc)

    def ApplyStructure(self):
        layer_msg = 'Initial resize convolution'
        #in_channel_mask = self.resnet_conv1.ApplyStructure(msg=layer_msg)
        in_channel_mask = None
        for i, cell in enumerate(self.cells):
            layer_msg = 'Cell {}'.format(i)
            out_channel_mask = cell.ApplyStructure(in1_channel_mask=in_channel_mask, msg=layer_msg)
            in_channel_mask = out_channel_mask

        self.fc.ApplyStructure(in_channel_mask=in_channel_mask)

    def forward(self, x, isTraining=False):
        #x = self.resnet_conv1(x)
        for i, cell in enumerate(self.cells):
            x = cell(x, isTraining=isTraining)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch dimension
        x = self.fc(x)
        return x

    def Cells(self):
        return self.cells

    def ArchitectureWeights(self):
        architecture_weights = []
        cell_weights = []

        #cell_archatecture_weights, cell_total_trainable_weights, conv_weights = self.resnet_conv1.ArchitectureWeights()
        #cell_weight = {'prune_weight':torch.tensor(1.0), 'cell_weight':[conv_weights]}
        #cell_weights.append(cell_weight)
        #architecture_weights.append(cell_archatecture_weights)

        for in_cell in self.cells:
            cell_archatecture_weights, cell_total_trainable_weights, cell_weight = in_cell.ArchitectureWeights()
            cell_weights.append(cell_weight)
            architecture_weights.append(cell_archatecture_weights)


        architecture_weights = torch.cat(architecture_weights)
        architecture_weights = torch.sum(architecture_weights)

        # Not yet minimizing fully connected layer weights
        # fc_archatecture_weights, fc_total_trainable_weights = self.fc.ArchitectureWeights()
        #archatecture_weights += fc_archatecture_weights

        return architecture_weights, self.total_trainable_weights, cell_weights

    def Parameters(self):
        current_weights = model_weights(self)
        if self.total_trainable_weights > 0:
            remnent =  model_weights(self)/self.total_trainable_weights
        else:
            remnent = None

            
        return current_weights, self.total_trainable_weights, remnent

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return not(v==0)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')
    parser.add_argument('-description', type=str, default="", help='Training description to record with resuts')
    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')

    parser.add_argument('-prune', type=str2bool, default=False)
    parser.add_argument('-train', type=str2bool, default=True)
    parser.add_argument('-infer', type=str2bool, default=True)
    parser.add_argument('-search_structure', type=str2bool, default=True)
    parser.add_argument('-onnx', type=str2bool, default=True)
    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152, 20, 32, 44, 56, 110], default=56, help='Run description')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('-batch_size', type=int, default=400, help='Training batch size') 

    parser.add_argument('-learning_rate', type=float, default=1e-1, help='Training learning rate')
    parser.add_argument('-learning_rate_decay', type=float, default=0.5, help='Rate decay multiple')
    parser.add_argument('-rate_schedule', type=json.loads, default='[50, 100, 150, 200, 250, 300, 350, 400, 450, 500]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[40, 60, 65]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[10, 15, 17]', help='Training learning rate')
    
    parser.add_argument('-momentum', type=float, default=0.9, help='Learning Momentum')
    parser.add_argument('-weight_decay', type=float, default=0.0001)

    parser.add_argument('-model_type', type=str,  default='Classification')
    parser.add_argument('-model_class', type=str,  default='CIFAR10')
    parser.add_argument('-model_src', type=str,  default=None)
    parser.add_argument('-model_dest', type=str, default="crisp20220511_t00_00")
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-k_structure', type=float, default=1e0, help='Structure minimization weighting factor')
    parser.add_argument('-target_structure', type=float, default=0.00, help='Structure minimization weighting factor')

    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.5, help='tanh pruning threshold')

    parser.add_argument('-augment_rotation', type=float, default=0.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_min', type=float, default=1.00, help='Input augmentation scale')
    parser.add_argument('-augment_scale_max', type=float, default=1.00, help='Input augmentation scale')
    parser.add_argument('-augment_translate_x', type=float, default=0.125, help='Input augmentation translation')
    parser.add_argument('-augment_translate_y', type=float, default=0.125, help='Input augmentation translation')
    parser.add_argument('-augment_noise', type=float, default=0.1, help='Augment image noise')

    parser.add_argument('-resultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='./tb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')



    args = parser.parse_args()
    return args

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model, out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)

class PlotSearch():
    def __init__(self, title = 'Architecture Weights', colormapname = 'jet', lenght = 5, dpi=1200, thickness=1 ):
        self.title = title
        self.colormapname = colormapname
        self.lenght = int(lenght)
        self.dpi = dpi
        self.cm = plt.get_cmap(colormapname)
        self.thickness=thickness

    def plot(self, weights, index = None):
        height = 0
        width = 0
        for i,  cell, in enumerate(weights):
            for j, step in enumerate(cell['cell_weight']):
                width += self.lenght
                height = max(height, self.thickness*len(step))


        img = np.zeros([height,width,3]).astype(np.uint8)
       
        #self.ax.clear()
        
        if index:
            title = '{} {}'.format(self.title, index)
        else:
            title = self.title        
        x = 0
        for i,  cell, in enumerate(weights):
            prune_weight = cell['prune_weight'].item()
            for j, step in enumerate(cell['cell_weight']):
                for k, gain in enumerate(step.cpu().detach().numpy()):
                    
                    y = int(k*self.thickness)
                    start_point = (x,y)
                    end_point=(x+self.lenght-1,y)

                    conv_gain = prune_weight*gain
                    color = 255*np.array(self.cm(conv_gain))
                    color = color.astype('uint8')
                    colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                    cv2.line(img,start_point,end_point,colorbgr,self.thickness)
                x += self.lenght

        #cv2.putText(img,title,(int(0.25*self.width), 30), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))

        return img


class PlotWeights():
    def __init__(self, title = 'Conv Norm', colormapname = 'jet', lenght = 5, dpi=1200, thickness=1, classification=True, pruning=True ):

        self.title = title
        self.colormapname = colormapname
        self.lenght = int(lenght)
        self.dpi = dpi
        self.cm = plt.get_cmap(colormapname)
        self.thickness=thickness
        self.classification=classification
        self.pruning=pruning

    def plot(self, network, index = None): 
        
        if index:
            title = '{} {}'.format(self.title, index)
        else:
            title = self.title

        weight_norms = []
        max_weight =  float('-inf')
        min_weight = float('inf')
        max_layers = 0
        for i,  cell, in enumerate(network.cells):
            if cell.cnn is not None:
                for j, convbr in enumerate(cell.cnn):
                    layer_norms = []
                    if convbr.conv.weight is not None:
                        if convbr.conv_transpose:
                            weights = torch.linalg.norm(convbr.conv.weight , dim=(0,2,3))/np.sqrt(np.product(convbr.conv.weight.shape[1:]))
                        else:
                            weights = torch.linalg.norm(convbr.conv.weight, dim=(1,2,3))/np.sqrt(np.product(convbr.conv.weight.shape[0:]))
                        numSums = 1                    

                        if self.pruning:
                            if convbr.channel_scale.grad is not None:
                                weights += torch.abs(convbr.channel_scale.grad)
                                numSums += 1

                        weights /= (numSums)

                        #x = i*self.lenght*len(cell.cnn)+j*self.lenght

                        for k, gradient_norm in enumerate(grads.cpu().detach().numpy()):
                            if gradient_norm > max_weight:
                                max_weight = gradient_norm
                            if gradient_norm < min_weight:
                                min_weight = gradient_norm

                            layer_norms.append(gradient_norm)
                            
                            '''y = int(k*self.thickness+self.thickness/2)
                            start_point = (x,y)
                            end_point=(x+self.lenght,y)

                            color = 255*np.array(self.cm(gradient_norm))
                            color = color.astype('uint8')
                            colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                            cv2.line(img,start_point,end_point,colorbgr,self.thickness)'''
                    weight_norms.append(layer_norms)
                    max_layers = max(max_layers, len(layer_norms))

        width = len(weight_norms)*self.lenght
        height = max_layers*self.thickness
        img = np.zeros([height,width,3]).astype(np.uint8)

        x = 0
        for j, weight_norm in enumerate(weight_norms):
            for k, gradient in enumerate(weight_norm):
                y = int(k*self.thickness)
                start_point = (x,y)
                end_point=(x+self.lenght-1,y)

                gain = (gradient-min_weight)/(max_weight-min_weight)
                color = 255*np.array(self.cm(gain))
                color = color.astype('uint8')
                colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                cv2.line(img,start_point,end_point,colorbgr,self.thickness)
            x += self.lenght

        grad_mag = '{:0.3e}'.format(max_weight, min_weight)
        cv2.putText(img,grad_mag,(int(0.05*width), int(0.90*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.75, color=(0, 255, 255))

        return img

class PlotGradients():
    def __init__(self, title = 'Gradient Norm', colormapname = 'jet', lenght = 5, dpi=1200, thickness=1, classification=True, pruning=True ):

        self.title = title
        self.colormapname = colormapname
        self.lenght = int(lenght)
        self.dpi = dpi
        self.cm = plt.get_cmap(colormapname)
        self.thickness=thickness
        self.classification=classification
        self.pruning=pruning

    def plot(self, network, index = None): 
        
        if index:
            title = '{} {}'.format(self.title, index)
        else:
            title = self.title

        gradient_norms = []
        max_gradient =  float('-inf')
        min_gradient = float('inf')
        max_layers = 0
        for i,  cell, in enumerate(network.cells):
            if cell.cnn is not None:
                for j, convbr in enumerate(cell.cnn):
                    layer_norms = []
                    if convbr.conv.weight.grad is not None:
                        if convbr.conv_transpose:
                            grads = torch.linalg.norm(convbr.conv.weight.grad , dim=(0,2,3))/np.sqrt(np.product(convbr.conv.weight.grad.shape[1:]))
                        else:
                            grads = torch.linalg.norm(convbr.conv.weight.grad , dim=(1,2,3))/np.sqrt(np.product(convbr.conv.weight.grad.shape[0:]))
                        numSums = 1                    

                        if self.pruning:
                            if convbr.channel_scale.grad is not None:
                                grads += torch.abs(convbr.channel_scale.grad)
                                numSums += 1

                        grads /= (numSums)

                        #x = i*self.lenght*len(cell.cnn)+j*self.lenght

                        for k, gradient_norm in enumerate(grads.cpu().detach().numpy()):
                            if gradient_norm > max_gradient:
                                max_gradient = gradient_norm
                            if gradient_norm < min_gradient:
                                min_gradient = gradient_norm

                            layer_norms.append(gradient_norm)
                            
                            '''y = int(k*self.thickness+self.thickness/2)
                            start_point = (x,y)
                            end_point=(x+self.lenght,y)

                            color = 255*np.array(self.cm(gradient_norm))
                            color = color.astype('uint8')
                            colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                            cv2.line(img,start_point,end_point,colorbgr,self.thickness)'''
                    gradient_norms.append(layer_norms)
                    max_layers = max(max_layers, len(layer_norms))

        width = len(gradient_norms)*self.lenght
        height = max_layers*self.thickness
        img = np.zeros([height,width,3]).astype(np.uint8)

        x = 0
        for j, gradient_norm in enumerate(gradient_norms):
            for k, gradient in enumerate(gradient_norm):
                y = int(k*self.thickness)
                start_point = (x,y)
                end_point=(x+self.lenght-1,y)

                gain = (gradient-min_gradient)/(max_gradient-min_gradient)
                color = 255*np.array(self.cm(gain))
                color = color.astype('uint8')
                colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                cv2.line(img,start_point,end_point,colorbgr,self.thickness)
            x += self.lenght

        grad_mag = '{:0.3e}'.format(max_gradient, min_gradient)
        cv2.putText(img,grad_mag,(int(0.05*width), int(0.90*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.75, color=(0, 255, 255))

        return img




class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    import torchvision
    import torchvision.transforms as transforms

    test_results={}
    test_results['system'] = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'OpenCV': cv2.__version__,
    }
    print('Cell Test system={}'.format(test_results['system'] ))

    test_results['args'] = {}
    for arg in vars(args):
        #log_param(arg, getattr(args, arg))
        test_results['args'][arg]=getattr(args, arg)

    print('arguments={}'.format(test_results['args']))

    torch.autograd.set_detect_anomaly(True)

    s3, creds, s3def = Connect(args.credentails)

    test_results['classes'] = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)

    device = "cpu"
    pin_memory = False
    if args.cuda:
        device = torch.device('cuda')
        pin_memory = True

    # Load dataset
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=args.augment_rotation, 
            translate=(args.augment_translate_x, args.augment_translate_y), 
            scale=(args.augment_scale_min, args.augment_scale_max), 
            interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), 
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), # Imagenet mean and standard deviation
        AddGaussianNoise(0., args.augment_noise)
    ])

    test_transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) # Imagenet mean and standard deviation
    ])

    trainingset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=train_transform)
    train_batches=int(trainingset.__len__()/args.batch_size)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=test_transform)
    test_batches=int(testset.__len__()/args.batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_freq = int(math.ceil(train_batches/test_batches))

    # Load dataset
    device = torch.device("cpu")
    pin_memory = False
    if args.cuda:
        device = torch.device("cuda")
        pin_memory = True
        
    # Create classifiers
    # Create Default classifier
    resnetCells = ResnetCells(Resnet(args.resnet_len))
    classify = Classify(convolutions=resnetCells, 
                        device=device, 
                        weight_gain=args.weight_gain, 
                        dropout_rate=args.dropout_rate, 
                        search_structure=args.search_structure, 
                        sigmoid_scale=args.sigmoid_scale,
                        batch_norm = args.batch_norm,
                        feature_threshold = args.feature_threshold
                        )

    total_parameters = count_parameters(classify)

    if(args.model_src and args.model_src != ''):
        #modeldict = s3.GetDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            classify = torch.load(io.BytesIO(modelObj))
            #classify.load_state_dict(saved_model.model.state_dict())
        else:
            print('Failed to load model {}. Exiting.'.format(args.model_src))
            return -1

    if args.prune:
        classify.ApplyStructure()
        reduced_parameters = count_parameters(classify)
        print('Reduced parameters {}/{} = {}'.format(reduced_parameters, total_parameters, reduced_parameters/total_parameters))
        save(classify, s3, s3def, args)

    # Enable multi-gpu processing
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(classify)
        classify = model.module
    else:
        model = classify

    #specificy device for model
    classify.to(device)



    # Define a Loss function and optimizer
    target_structure = torch.as_tensor([args.target_structure], dtype=torch.float32)
    criterion = TotalLoss(args.cuda, k_structure=args.k_structure, target_structure=target_structure, search_structure=args.search_structure)
    optimizer = optim.SGD(classify.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay )
    #optimizer = optim.Adam(classify.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay )
    plotsearch = PlotSearch()
    plotgrads = PlotGradients()
    #scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.rate_schedule, gamma=args.learning_rate_decay)
    iSample = 0

    # Train
    test_results['train'] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'architecture_reduction':[]}
    test_results['test'] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'architecture_reduction':[], 'accuracy':[]}
    if args.train:
        for epoch in tqdm(range(args.epochs), desc="Train epochs", disable=args.job):  # loop over the dataset multiple times
            iTest = iter(testloader)

            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader), total=trainingset.__len__()/args.batch_size, desc="Train steps", disable=args.job):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs, isTraining=True)
                loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale  = criterion(outputs, labels, classify)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if writer is not None:
                    writer.add_scalar('loss/train', loss, iSample)
                    writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, iSample)
                    writer.add_scalar('CRISP/architecture_loss', architecture_loss, iSample)
                    writer.add_scalar('CRISP/prune_loss', prune_loss, iSample)
                    writer.add_scalar('CRISP/architecture_reduction', architecture_reduction, iSample)
                #test_results['train']['loss'].append(loss.item())
                #test_results['train']['cross_entropy_loss'].append(cross_entropy_loss.item())
                #test_results['train']['architecture_loss'].append(architecture_loss.item())
                #test_results['train']['architecture_reduction'].append(architecture_reduction.item())

                if i % test_freq == test_freq-1:    # Save image and run test

                    classifications = torch.argmax(outputs, 1)
                    results = (classifications == labels).float()
                    training_accuracy = torch.sum(results)/len(results)

                    data = next(iTest)
                    inputs, labels = data

                    if args.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    #optimizer.zero_grad()
                    outputs = model(inputs)
                    classifications = torch.argmax(outputs, 1)
                    results = (classifications == labels).float()
                    test_accuracy = torch.sum(results)/len(results)

                    loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale  = criterion(outputs, labels, classify)

                    running_loss /=test_freq
                    msg = '[{:3}/{}, {:6d}/{}]  accuracy: {:05f}|{:05f} loss: {:0.5e}|{:0.5e} remaining: {:0.5e} (train|test)'.format(
                        epoch + 1,
                        args.epochs, 
                        i + 1, 
                        trainingset.__len__()/args.batch_size, 
                        training_accuracy.item(), 
                        test_accuracy.item(), 
                        running_loss, 
                        loss.item(), 
                        architecture_reduction.item()
                    )
                    if args.job is True:
                        print(msg)
                    else:
                        tqdm.write(msg)
                    running_loss = 0.0

                    if writer is not None:
                        writer.add_scalar('accuracy/train', training_accuracy, iSample)
                        writer.add_scalar('accuracy/test', test_accuracy, iSample)
                        writer.add_scalar('loss/test', loss, iSample)
                        writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, iSample)
                        writer.add_scalar('CRISP/architecture_loss', architecture_loss, iSample)
                        writer.add_scalar('CRISP/prune_loss', prune_loss, iSample)
                        writer.add_scalar('CRISP/architecture_reduction', architecture_reduction, iSample)

                    #test_results['train']['loss'].append(running_loss)
                    #test_results['test']['loss'].append(loss)
                    #test_results['test']['cross_entropy_loss'].append(cross_entropy_loss.item())
                    #test_results['test']['architecture_loss'].append(architecture_loss.item())
                    #test_results['test']['architecture_reduction'].append(architecture_reduction.item())
                    #test_results['test']['accuracy'].append(test_accuracy.item())

                    #log_metric("accuracy", test_accuracy.item())

                #iSave = 2000
                #if i % iSave == iSave-1:    # print every iSave mini-batches
                #    save(classify, s3, s3def, args)

                if args.fast and i+1 >= test_freq:
                    break

                iSample += 1

            cv2.imwrite('class_weights.png', plotsearch.plot(cell_weights))
            cv2.imwrite('gradient_norm.png', plotgrads.plot(classify))

            #scheduler1.step()
            scheduler2.step()
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            img = plotsearch.plot(cell_weights)
            is_success, buffer = cv2.imencode(".png", img, compression_params)
            img_enc = io.BytesIO(buffer).read()
            filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
            s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

            # Plot gradients before saving which clears the gradients
            img = plotgrads.plot(classify)
            is_success, buffer = cv2.imencode(".png", img)  
            img_enc = io.BytesIO(buffer).read()
            filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
            s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

            save(classify, s3, s3def, args)

        torch.cuda.empty_cache()
        accuracy = 0.0
        iTest = iter(testloader)
        for i, data in tqdm(enumerate(testloader), total=testset.__len__()/args.batch_size, desc="Test steps"):
            inputs, labels = data

            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            with torch.no_grad():
                outputs = classify(inputs)

            classifications = torch.argmax(outputs, 1)
            results = (classifications == labels).float()
            accuracy += torch.sum(results).item()

        accuracy /= args.batch_size*int(testset.__len__()/args.batch_size)
        #log_metric("test_accuracy", accuracy)
        print("test_accuracy={}".format(accuracy))
        test_results['test_accuracy'] = accuracy
        test_results['current_weights'], test_results['original_weights'], test_results['remnent'] = classify.Parameters()

        s3.PutDict(s3def['sets']['model']['bucket'], '{}/{}/{}_results.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), test_results)

    if args.resultspath is not None and len(args.resultspath) > 0:
        WriteDictJson(test_results, args.resultspath)

    print('Finished {}  {}'.format(args.model_dest, test_results))
    return 0

if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        ''' 
        https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        Add a "Python: Remote" configuraiton to launch.json:
        "configurations": [
            {
                "name": "Python: Remote",
                "type": "python",
                "request": "attach",
                "port": 3000,
                "host": "localhost",
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "."
                    }
                ],
                "justMyCode": false
            },
            ...

        Launch application from console with -debug flag
        $ python3 train.py -debug

        Connet to vscode "Python: Remote" configuration
        '''

        debugpy.listen(address=(args.debug_address, args.debug_port))
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = Test(args)
    sys.exit(result)

