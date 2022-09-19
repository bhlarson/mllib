#!/usr/bin/python3
import math
import os
import sys
import io
import json
import yaml
import platform
from enum import Enum
import time
from datetime import datetime
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard import program
import torch.profiler
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDict, WriteDict, str2bool
from pymlutil.s3 import s3store, Connect
from pymlutil.functions import Exponential, GaussianBasis
from pymlutil.metrics import DatasetResults
import pymlutil.version as pymlutil_version
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torchdatasetutil.imagenetstore import CreateImagesetLoaders
from torchdatasetutil.cifar10store import CreateCifar10Loaders

from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, os.path.abspath(''))
from networks.totalloss import TotalLoss, FenceSitterEjectors

# Inner neural architecture cell repetition structure
# Process: Con2d, optional batch norm, optional ReLu

def relu_flops_counter_hook(module, shape, relaxation):
    flops_total = np.prod(shape)
    flops_relaxed = flops_total

    if relaxation is not None:
        channel_weight = relaxation.weights()
        num_channels = len(channel_weight)
        flops_relaxed *= torch.sum(channel_weight)/num_channels

    return flops_relaxed, flops_total

def bn_flops_counter_hook(module, shape, relaxation):
    flops_total = np.prod(shape)
    if module.affine:
        flops_total *= 2

    flops_relaxed = flops_total

    if relaxation is not None:
        channel_weight = relaxation.weights()
        num_channels = len(channel_weight)
        flops_relaxed *= torch.sum(channel_weight)/num_channels

    return flops_relaxed, flops_total

def conv_flops_counter_hook(conv_module, in_relaxation, output_shape, out_relaxation):
    # Can have multiple inputs, getting the first one
    output_dims = list(output_shape)[2:]

    kernel_dims = list(conv_module.kernel_size)
    #in_channels = conv_module.in_channels
    input_relaxation = []
    if in_relaxation is not None and len(in_relaxation) > 0:
        for in_relaxation_source in in_relaxation:
            input_relaxation.append(in_relaxation_source.weights())

    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    if len(input_relaxation) > 0:
        in_channel_weight = torch.sum(torch.cat(input_relaxation, 0))
    else:
        in_channel_weight = in_channels

    if out_relaxation is not None:
        out_channel_weight = torch.sum(out_relaxation.weights())
    else:
        out_channel_weight = out_channels

    active_elements_count = int(np.prod(output_dims))

    if conv_module.bias is not None:
        bias_flops = out_channel_weight * active_elements_count
    else:
        bias_flops = 0

    conv_base = active_elements_count * int(np.prod(kernel_dims)) / groups
    conv_flops_relaxed = conv_base * in_channel_weight * out_channel_weight + bias_flops
    conv_flops_total = conv_base * in_channels * out_channels + bias_flops

    return conv_flops_relaxed, conv_flops_total

class RelaxChannels(nn.Module):
    def __init__(self,channels, 
                 device=torch.device("cpu"),
                 search_structure=True,
                 disable_search_structure = False,
                 sigmoid_scale = 5.0, # Channel sigmoid scale factor
                ):
        super(RelaxChannels, self).__init__()
        self.channels = channels
        self.device = device
        self.search_structure = search_structure
        self.disable_search_structure = disable_search_structure
        self.sigmoid_scale = sigmoid_scale
        self.channel_scale = nn.Parameter(torch.zeros(self.channels, dtype=torch.float, device=self.device))

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.ones_(self.channel_scale)

    def ApplyParameters(self, search_structure=None, sigmoid_scale=None):
        if search_structure is not None:
            self.disable_search_structure = not search_structure
        if sigmoid_scale is not None:
            self.sigmoid_scale = sigmoid_scale

    def weights(self):
        conv_weights = self.sigmoid(self.sigmoid_scale*self.channel_scale)
        return conv_weights

    def forward(self, x):
        # if self.search_structure and not self.disable_search_structure: #scale channels based on self.channel_scale
        #         x *= self.weights()[None,:,None,None]
        x *= self.weights()[None,:,None,None]
        return x

    # Remove specific network dimensions
    # remove dimension where inDimensions and outDimensions arrays are 0 for channels to be removed
    def ApplyStructure(self, conv_mask, msg=None):
        if msg is None:
            prefix = "RelaxChannels::ApplyStructure"
        else:
            prefix = "RelaxChannels::ApplyStructure {}".format(msg)

        self.channel_scale.data = self.channel_scale.data[conv_mask!=0]

        print("{} channel_scale.shape()={}".format(prefix, self.channel_scale.shape))

        return conv_mask

class ConvBR(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 prev_relaxation = None,
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
                 disable_search_structure = False,
                 search_flops = False,
                 ):
        super(ConvBR, self).__init__()
        self.in_channels = in_channels
        if out_channels < 1:
            raise ValueError("out_channels must be > 0")
        self.out_channels = out_channels
        self.prev_relaxation = prev_relaxation
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
        self.disable_search_structure = disable_search_structure
        self.residual = residual
        self.use_dropout = dropout
        self.conv_transpose = conv_transpose
        self.k_prune_sigma=k_prune_sigma
        self.device = device
        self.search_flops = search_flops
        self.input_shape = None
        self.output_shape = None

        #self.channel_scale = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float, device=self.device))

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

        self.total_trainable_weights = model_weights(self)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

        if self.relu:
            self.activation = nn.ReLU()
        else:
            self.activation = None

        if self.search_structure:
            self.relaxation = RelaxChannels(self.out_channels, 
                                            device=self.device, 
                                            search_structure=self.search_structure,
                                            disable_search_structure=self.disable_search_structure,
                                            sigmoid_scale=self.sigmoid_scale
                                            )
        else:
            self.relaxation = None

        self._initialize_weights()

    def _initialize_weights(self):
        #nn.init.normal_(self.channel_scale, mean=0.5,std=0.33)
        #nn.init.ones_(self.channel_scale)
        #nn.init.zeros_(self.channel_scale)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.batch_norm and self.batchnorm2d and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def ApplyParameters(self, search_structure=None, convMaskThreshold=None, dropout=None,
                        sigmoid_scale=None, weight_gain=None, k_prune_sigma=None, search_flops=None):
        if search_structure is not None:
            self.disable_search_structure = not search_structure
        if convMaskThreshold is not None:
            self.convMaskThreshold = convMaskThreshold
        if dropout is not None:
            self.use_dropout = dropout
        if sigmoid_scale is not None:
            self.sigmoid_scale = sigmoid_scale
        if weight_gain is not None:
            self.weight_gain = weight_gain
        if k_prune_sigma is not None:
            self.k_prune_sigma = k_prune_sigma
        if search_flops is not None:
            self.search_flops = search_flops

        if self.relaxation is not None:
            self.relaxation.ApplyParameters(search_structure, sigmoid_scale)

    def forward(self, x):

        if self.input_shape is None:
            self.input_shape = x.shape

        if self.out_channels > 0:
            if self.use_dropout:
                x = self.dropout(x)
                
            x = self.conv(x)
  
            if self.batch_norm:
                x = self.batchnorm2d(x)

            if self.relaxation:
                x = self.relaxation(x)  

            if self.activation:
                x = self.activation(x)

        else :
            print("Failed to prune zero size convolution")

        if self.output_shape is None:
            self.output_shape = x.shape

        return x

    def ArchitectureWeights(self):

        if self.relaxation and self.search_structure and not self.disable_search_structure:
            weight_basis = GaussianBasis(self.relaxation.weights(), sigma=self.k_prune_sigma)
            conv_weights = self.relaxation.weights()

        else:
            weight_basis = torch.zeros((self.out_channels), device=self.device)
            conv_weights = torch.ones((self.out_channels), device=self.device)


        if self.out_channels == 0:
            architecture_weights = conv_weights.sum_to_size((1))
        else:

            if self.search_flops:
                architecture_weights, cell_weights = conv_flops_counter_hook(self.conv, self.prev_relaxation, self.output_shape, self.relaxation)

                if self.activation:
                    flops_relaxed, flops_total = relu_flops_counter_hook(self.activation, self.output_shape, self.relaxation)
                    architecture_weights += flops_relaxed
                    cell_weights += flops_total
                if self.batch_norm:
                    flops_relaxed, flops_total = bn_flops_counter_hook(self.batchnorm2d, self.output_shape, self.relaxation)
                    architecture_weights += flops_relaxed
                    cell_weights += flops_total

                if not torch.is_tensor(architecture_weights):
                    architecture_weights = torch.tensor(architecture_weights, device = self.device)

                architecture_weights = torch.reshape(architecture_weights,[-1]) # reshape to single element array to be the same format as not flops architecture_weights

            else:
                cell_weights = model_weights(self)

                # Keep sum as [1] tensor so subsequent concatenation works
                architecture_weights = (cell_weights/ self.out_channels) * conv_weights.sum_to_size((1))


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
        elif self.relaxation and self.search_structure and not self.disable_search_structure:
            conv_mask = self.relaxation.weights() > self.convMaskThreshold
        else:
            conv_mask = torch.ones((self.out_channels), dtype=torch.bool, device=self.device) # Always true if self.search_structure == False

        if out_channel_mask is not None:
            if len(out_channel_mask) == self.out_channels:
                # If either mask is false, then the convolution is removed (not nand)
                conv_mask = torch.logical_not(torch.logical_or(torch.logical_not(conv_mask), torch.logical_not(out_channel_mask)))
                
            else:
                raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))

        prev_convolutions = len(conv_mask)
        pruned_convolutions = len(conv_mask[conv_mask==False])

        if self.residual and prev_convolutions==pruned_convolutions: # Pruning full convolution 
            conv_mask = ~conv_mask # Residual connection becomes a straight through connection
        else:
            self.conv.bias.data = self.conv.bias[conv_mask!=0]
            if self.conv_transpose:
                self.conv.weight.data = self.conv.weight[:, conv_mask!=0]
            else:
                self.conv.weight.data = self.conv.weight[conv_mask!=0]

            #self.channel_scale.data = self.channel_scale.data[conv_mask!=0]

            if self.batch_norm:
                self.batchnorm2d.bias.data = self.batchnorm2d.bias.data[conv_mask!=0]
                self.batchnorm2d.weight.data = self.batchnorm2d.weight.data[conv_mask!=0]
                self.batchnorm2d.running_mean = self.batchnorm2d.running_mean[conv_mask!=0]
                self.batchnorm2d.running_var = self.batchnorm2d.running_var[conv_mask!=0]

            if self.relaxation and self.search_structure and not self.disable_search_structure:
                self.relaxation.ApplyStructure(conv_mask)

            self.out_channels = len(conv_mask[conv_mask!=0])

        print("{} {}={}/{} in_channels={} out_channels={}".format(prefix, pruned_convolutions/prev_convolutions, pruned_convolutions, prev_convolutions, self.in_channels, self.out_channels))

        return conv_mask


DefaultMaxDepth = 1
class Cell(nn.Module):
    def __init__(self,
                 in1_channels, 
                 in2_channels = 0,
                 prev_relaxation = None,
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
                 k_prune_sigma=0.33,
                 search_flops = False,
                 ):
                
        super(Cell, self).__init__()

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.prev_relaxation = prev_relaxation
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
        self.search_flops = search_flops


        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convolutions uses out_channels as chanels

        src_channels = in_chanels = self.in1_channels+self.in2_channels

        totalStride = 1
        totalDilation = 1
        prev_relaxation = self.prev_relaxation

        for i, convdev in enumerate(convolutions):
            conv_transpose = False
            if 'conv_transpose' in convdev and convdev['conv_transpose']:
                conv_transpose = True

            conv = ConvBR(src_channels, convdev['out_channels'], 
                prev_relaxation = prev_relaxation,
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
                search_structure=convdev['search_structure'] and self.search_structure,
                sigmoid_scale = self.sigmoid_scale,
                residual=False, 
                dropout=self.dropout,
                conv_transpose=conv_transpose,
                k_prune_sigma=self.k_prune_sigma,
                device=self.device,
                search_flops = self.search_flops)

            prev_relaxation = [conv.relaxation]
            
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
                        k_prune_sigma=None, search_flops=None): # Apply a parameter change
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

        if search_flops is not None:
            self.search_flops = search_flops

        if self.cnn is not None and len(self.cnn) > 0:
            for conv in self.cnn:
                conv.ApplyParameters(search_structure=search_structure, convMaskThreshold=convMaskThreshold, dropout=dropout,
                                     weight_gain=weight_gain, sigmoid_scale=sigmoid_scale, k_prune_sigma=k_prune_sigma, search_flops=search_flops)

    def ArchitectureWeights(self):
        layer_weights = []
        conv_weights = []
        weight_basises = []
        norm_conv_weight = []
        search_structure = []
        model_weight_total = 0
        unallocated_weights  = torch.zeros((1), device=self.device)
        architecture_weights = torch.zeros((1), device=self.device)
        if self.cnn is not None:
            for i, l in enumerate(self.cnn): 
                layer_weight, cnn_weight, conv_weight, weight_basis = l.ArchitectureWeights()
                model_weight_total += int(cnn_weight)
                conv_weights.append(conv_weight)
                weight_basises.append(weight_basis)

                if self.convolutions[i]['search_structure']:
                    architecture_weights += layer_weight[0]
                    norm_conv_weight.append(layer_weight/cnn_weight)
                else:
                    unallocated_weights += cnn_weight

            if architecture_weights[0] > 0:
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

        return architecture_weights, model_weight_total, cell_weights

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
                 device=torch.device("cpu"),
                 ):
        super(FC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device


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
    def __init__(self, 
                convolutions, 
                device=torch.device("cpu"), 
                source_channels = 3, 
                out_channels = 10, 
                initial_channels=16, 
                batch_norm=True, 
                weight_gain=11, 
                convMaskThreshold=0.5, 
                dropout_rate=0.2, 
                search_structure = True, 
                sigmoid_scale=5.0, 
                feature_threshold=0.5, 
                search_flops = True,):
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
        self.search_flops = search_flops
                
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
                feature_threshold=self.feature_threshold,
                             search_flops = self.search_flops)
            in_channels = cell_convolutions['cell'][-1]['out_channels']
            self.cells.append(cell)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FC(in_channels, self.out_channels, device=self.device)

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

    def ApplyParameters(self, search_structure=None, convMaskThreshold=None, dropout=None, 
                        weight_gain=None, sigmoid_scale=None, feature_threshold=None,
                        k_prune_sigma=None, search_flops=None): # Apply a parameter change
        if search_structure is not None:
            self.search_structure = search_structure
        if dropout is not None:
            self.use_dropout = dropout
        if convMaskThreshold is not None:
            self.convMaskThreshold = convMaskThreshold
        if weight_gain is not None:
            self.weight_gain = weight_gain
        if sigmoid_scale is not None:
            self.sigmoid_scale = sigmoid_scale
        if feature_threshold is not None:
            self.feature_threshold = feature_threshold
        if k_prune_sigma is not None:
            self.k_prune_sigma = k_prune_sigma
        if search_flops is not None:
            self.search_flops = search_flops
        for cell in self.cells:
            cell.ApplyParameters(search_structure=search_structure, dropout=dropout, convMaskThreshold=convMaskThreshold,
                                 weight_gain=weight_gain, sigmoid_scale=sigmoid_scale, feature_threshold=feature_threshold,
                                 k_prune_sigma=k_prune_sigma, search_flops=search_flops)

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

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug', type=str2bool, default=False, help='Wait for debuggee attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-min', action='store_true', help='Minimum run with a few iterations to test execution')
    parser.add_argument('-minimum', type=str2bool, default=False, help='Minimum run with a few iterations to test execution')

    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152, 20, 32, 44, 56, 110], default=101, help='Run description')

    parser.add_argument('-dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'], help='Dataset')
    parser.add_argument('-dataset_path', type=str, default='/data', help='Local dataset path')
    parser.add_argument('-obj_imagenet', type=str, default='data/imagenet', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-batch_size', type=int, default=80, help='Training batch size') 

    parser.add_argument('-learning_rate', type=float, default=1e-1, help='Training learning rate')
    parser.add_argument('-learning_rate_decay', type=float, default=0.5, help='Rate decay multiple')
    parser.add_argument('-rate_schedule', type=json.loads, default='[50, 100, 150, 200, 250, 300, 350, 400, 450, 500]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[40, 60, 65]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[10, 15, 17]', help='Training learning rate')
    
    parser.add_argument('-momentum', type=float, default=0.9, help='Learning Momentum')
    parser.add_argument('-weight_decay', type=float, default=0.0001)
    parser.add_argument('-epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('-start_epoch', type=int, default=0, help='Start epoch')

    parser.add_argument('-num_workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('-model_type', type=str,  default='classification')
    parser.add_argument('-model_class', type=str,  default='imagenet')
    parser.add_argument('-model_src', type=str,  default=None)
    parser.add_argument('-model_dest', type=str, default="crisp20220511_t00_00")
    parser.add_argument('-test_sparsity', type=int, default=1, help='test step multiple')
    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-cuda', type=bool, default=True)

    parser.add_argument('-height', type=int, default=200, help='Input image height')
    parser.add_argument('-width', type=int, default=200, help='Input image width')
    parser.add_argument('-channels', type=int, default=3, help='Input image color channels')
    parser.add_argument('-k_accuracy', type=float, default=1.0, help='Accuracy weighting factor')
    parser.add_argument('-k_structure', type=float, default=0.5, help='Structure minimization weighting factor')
    parser.add_argument('-k_prune_basis', type=float, default=1.0, help='prune base loss scaling')
    parser.add_argument('-k_prune_exp', type=float, default=50.0, help='prune basis exponential weighting factor')
    parser.add_argument('-k_prune_sigma', type=float, default=1.0, help='prune basis exponential weighting factor')
    parser.add_argument('-target_structure', type=float, default=0.00, help='Structure minimization weighting factor')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.5, help='tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.1, help='convolution channel sigmoid level to prune convolution channels')

    parser.add_argument('-augment_rotation', type=float, default=0.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_min', type=float, default=1.00, help='Input augmentation scale')
    parser.add_argument('-augment_scale_max', type=float, default=1.00, help='Input augmentation scale')
    parser.add_argument('-augment_translate_x', type=float, default=0.125, help='Input augmentation translation')
    parser.add_argument('-augment_translate_y', type=float, default=0.125, help='Input augmentation translation')
    parser.add_argument('-augment_noise', type=float, default=0.1, help='Augment image noise')

    parser.add_argument('-ejector', type=FenceSitterEjectors, default=FenceSitterEjectors.prune_basis, choices=list(FenceSitterEjectors))
    parser.add_argument('-ejector_start', type=float, default=4, help='Ejector start epoch')
    parser.add_argument('-ejector_full', type=float, default=5, help='Ejector full epoch')
    parser.add_argument('-ejector_max', type=float, default=1.0, help='Ejector max value')
    parser.add_argument('-ejector_exp', type=float, default=3.0, help='Ejector exponent')
    parser.add_argument('-prune', type=str2bool, default=True)
    parser.add_argument('-train', type=str2bool, default=True)
    parser.add_argument('-test', type=str2bool, default=True)
    parser.add_argument('-search_structure', type=str2bool, default=True)
    parser.add_argument('-search_flops', type=str2bool, default=True)
    parser.add_argument('-profile', type=str2bool, default=False)
    parser.add_argument('-time_trial', type=str2bool, default=False)
    parser.add_argument('-onnx', type=str2bool, default=True)
    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-resultspath', type=str, default='results.yaml')
    parser.add_argument('-prevresultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default=None)
    #parser.add_argument('-tensorboard_dir', type=str, default='/tb_logs',
    parser.add_argument('-tensorboard_dir', type=str, default=None,
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    parser.add_argument('-tb_dest', type=str, default='crispcifar10_20220909_061234_hiocnn_tb_01')

    parser.add_argument('-description', type=json.loads, default='{"description":"CRISP classificaiton"}', help='Test description')

    args = parser.parse_args()

    if args.d:
        args.debug = args.d
    if args.min:
        args.minimum = args.min

    if args.dataset == 'cifar10':
        args.width = 32
        args.height = 32
        args.channels = 3

    return args

def ModelSize(args, model, results, loaders):

    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset))

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")

    input = torch.zeros((1, testloader['in_channels'], testloader['height'], testloader['width']), device=device)

    # flops, params = get_model_complexity_info(deepcopy(model), (class_dictionary['input_channels'], args.height, args.width), as_strings=False,
    #                                     print_per_layer_stat=True, verbose=False)
    flops = FlopCountAnalysis(model, input)
    parameters = count_parameters(model)
    image_flops = flops.total()

    return parameters, image_flops

def load(s3, s3def, args, loaders, results):
    model = None

    if 'initial_parameters' not in results or args.model_src is None or args.model_src == '':
        model = MakeNetwork(args, source_channels = loaders[0]['in_channels'], out_channels = loaders[0]['num_classes'])
        results['initial_parameters'] , results['initial_flops'] = ModelSize(args, model, results, loaders)

    print('load initial_parameters = {} initial_flops = {}'.format(results['initial_parameters'], results['initial_flops']))

    if(args.model_src and args.model_src != ''):
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            model = torch.load(io.BytesIO(modelObj))

            results['model_parameters'] , results['model_flops'] = ModelSize(args, model, results, loaders)

            print('load model_parameters = {} model_flops = {}'.format(results['model_parameters'], results['model_flops']))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
            return model

    return model, results

def save(model, s3, s3def, args, loc=''):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    #torch.save(model.state_dict(), out_buffer) # To save just state dictionary, need to determine pruned network from state dict
    torch.save(model, out_buffer)
    outname = '{}/{}/{}{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest,loc)
    s3.PutObject(s3def['sets']['model']['bucket'], outname, out_buffer)

def MakeNetwork(args, source_channels = 3, out_channels = 10):
    resnetCells = ResnetCells(Resnet(args.resnet_len))

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")
    network = Classify(convolutions=resnetCells, 
                        device=device, 
                        weight_gain=args.weight_gain, 
                        dropout_rate=args.dropout_rate, 
                        search_structure=args.search_structure, 
                        sigmoid_scale=args.sigmoid_scale,
                        batch_norm = args.batch_norm,
                        feature_threshold = args.feature_threshold, 
                        source_channels = source_channels, 
                        out_channels = out_channels, 
                        )

    network.to(device)
    return network


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
                            if convbr.relaxation.channel_scale.grad is not None:
                                weights += torch.abs(convbr.relaxation.channel_scale.grad)
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
                            grads = convbr.conv.weight.grad.permute(1, 0, 2, 3).flatten(1).norm(dim=1)/np.sqrt(np.product(convbr.conv.weight.grad.shape[1:]))
                        else:
                            grads = convbr.conv.weight.grad.flatten(1).norm(dim=1)/np.sqrt(np.product(convbr.conv.weight.grad.shape[0:]))
                        numSums = 1                    

                        if self.pruning:
                            if convbr.relaxation is not None and convbr.relaxation.channel_scale is not None and convbr.relaxation.channel_scale.grad is not None:
                                grads += torch.abs(convbr.relaxation.channel_scale.grad)
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

default_loaders = [{'set':'train', 'enable_transform':True},
                   {'set':'test', 'enable_transform':False}]

def Train(args, s3, s3def, model, loaders, device, results, writer, profile=None):

    trainloader = next(filter(lambda d: d.get('set') == 'train', loaders), None)
    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)

    if trainloader is None:
        raise ValueError('{} {} failed to load trainloader {}'.format(__file__, __name__, args.dataset)) 
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset))


    # Define a Loss function and optimizer
    target_structure = torch.as_tensor([args.target_structure], dtype=torch.float32, device=device)

    if args.search_flops:
        total_weights= results['initial_flops'] 
    else:
        total_weights= results['initial_parameters'] 
    loss_fcn = TotalLoss(args.cuda,
                            k_accuracy=args.k_accuracy,
                            k_structure=args.k_structure, 
                            target_structure=target_structure, 
                            search_structure=args.search_structure, 
                            k_prune_basis=args.k_prune_basis, 
                            k_prune_exp=args.k_prune_exp,
                            sigmoid_scale=args.sigmoid_scale,
                            ejector=args.ejector,
                            total_weights= total_weights,
                            )

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay )
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay )
    plotsearch = PlotSearch()
    plotgrads = PlotGradients()
    #scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.rate_schedule, gamma=args.learning_rate_decay)

    test_freq = args.test_sparsity*int(math.ceil(trainloader['batches']/testloader['batches']))
    tstart = None
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    # Train
    results['train'] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'architecture_reduction':[]}
    results['test'] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'architecture_reduction':[], 'accuracy':[]}
    # Set up fence sitter ejectors
    ejector_exp = None
    if args.ejector == FenceSitterEjectors.dais or args.ejector == FenceSitterEjectors.dais.value:
        writer.add_scalar('CRISP/sigmoid_scale', args.sigmoid_scale, results['batches'])
        if args.epochs > args.ejector_start and args.ejector_max > args.sigmoid_scale:
            ejector_exp =  Exponential(vx=args.ejector_start, vy=args.sigmoid_scale, px=args.ejector_full, py=args.ejector_max, power=args.ejector_exp)

    elif args.ejector == FenceSitterEjectors.prune_basis or args.ejector == FenceSitterEjectors.prune_basis.value:
        #writer.add_scalar('CRISP/k_prune_basis', args.k_prune_basis, results['batches'])
        if args.epochs > args.ejector_start and args.ejector_max > 0:
            ejector_exp =  Exponential(vx=args.ejector_start, vy=0, px=args.ejector_full, py=args.ejector_max, power=args.ejector_exp)

    write_graph = False
    for epoch in tqdm(range(args.start_epoch, args.epochs), 
                        bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}', 
                        desc="Train epochs", disable=args.job):  # loop over the dataset multiple times
        iTest = iter(testloader['dataloader'])

        if ejector_exp is not None:
            if (args.ejector == FenceSitterEjectors.dais or args.ejector == FenceSitterEjectors.dais.value):
                sigmoid_scale = ejector_exp.f(float(epoch)).item()
                model.ApplyParameters(sigmoid_scale=sigmoid_scale, k_prune_sigma=args.k_prune_sigma)
                writer.add_scalar('CRISP/sigmoid_scale', sigmoid_scale, results['batches'])
            elif args.ejector == FenceSitterEjectors.prune_basis or args.ejector == FenceSitterEjectors.prune_basis.value:
                loss_fcn.k_prune_basis = args.k_prune_basis*ejector_exp.f(float(epoch)).item()
            #writer.add_scalar('CRISP/k_prune_basis', loss_fcn.k_prune_basis, results['batches'])

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader['dataloader']), 
                            bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}', 
                            total=trainloader['batches'], desc="Train batches", disable=args.job):

            try:
                # get the inputs; data is a list of [inputs, labels]
                prevtstart = tstart
                tstart = time.perf_counter()

                inputs, labels = data

                if args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                if writer is not None and not write_graph:
                    writer.add_graph(model, inputs)
                    write_graph = True

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs, isTraining=True)
                classifications = torch.argmax(outputs, 1)
                tinfer = time.perf_counter()
                loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale  = loss_fcn(outputs, labels, model)
                tloss = time.perf_counter()
                loss.backward()
                optimizer.step()
                tend = time.perf_counter()

                top1_correct = (classifications == labels).float()
                training_accuracy = torch.sum(top1_correct)/len(top1_correct)

                dtInfer = tinfer - tstart
                dtLoss = tloss - tinfer
                dtBackprop = tend - tloss
                dtCompute = tend - tstart

                dtCycle = 0
                if prevtstart is not None:
                    dtCycle = tstart - prevtstart

                # print statistics
                running_loss += loss.item()
                training_cross_entropy_loss = cross_entropy_loss

                if writer is not None:
                    writer.add_scalar('loss/train', loss, results['batches'])
                    writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, results['batches'])
                    writer.add_scalar('accuracy/train', training_accuracy, results['batches'])
                    writer.add_scalar('time/infer', dtInfer, results['batches'])
                    writer.add_scalar('time/loss', dtLoss, results['batches'])
                    writer.add_scalar('time/backpropegation', dtBackprop, results['batches'])
                    writer.add_scalar('time/compute', dtCompute, results['batches'])
                    writer.add_scalar('time/cycle', dtCycle, results['batches'])
                    writer.add_scalar('CRISP/architecture_loss', architecture_loss, results['batches'])
                    writer.add_scalar('CRISP/prune_loss', prune_loss, results['batches'])
                    writer.add_scalar('CRISP/architecture_reduction', architecture_reduction, results['batches'])
                    #writer.add_scalar('CRISP/sigmoid_scale', sigmoid_scale, results['batches'])

                if i % test_freq == test_freq-1:    # Save image and run test
                    if writer is not None:
                        imprune_weights = plotsearch.plot(cell_weights)
                        if imprune_weights.size > 0:
                            im_class_weights = cv2.cvtColor(imprune_weights, cv2.COLOR_BGR2RGB)
                            writer.add_image('network/prune_weights', im_class_weights, 0,dataformats='HWC')

                        imgrad = plotgrads.plot(model)
                        if imgrad.size > 0:
                            im_grad_norm = cv2.cvtColor(imgrad, cv2.COLOR_BGR2RGB)
                            writer.add_image('network/gradient_norm', im_grad_norm, 0,dataformats='HWC')


                    classifications = torch.argmax(outputs, 1)
                    top1_correct = (classifications == labels).float()
                    training_accuracy = torch.sum(top1_correct)/len(top1_correct)
                    with torch.no_grad():
                        data = next(iTest)
                        inputs, labels = data

                        if args.cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                    #with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale = loss_fcn(outputs, labels, model)
                    classifications = torch.argmax(outputs, 1)
                    top1_correct = (classifications == labels).float()
                    test_accuracy = torch.sum(top1_correct)/len(top1_correct)

                    if writer is not None:
                        writer.add_scalar('loss/test', loss, results['batches'])
                        writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, results['batches'])
                        writer.add_scalar('accuracy/test', test_accuracy, results['batches'])

                    running_loss /=test_freq
                    msg = '[{:3}/{}, {:6d}/{}]  loss: {:0.5e}|{:0.5e} cross-entropy loss: {:0.5e}|{:0.5e} accuracy: {:0.5e}|{:0.5e} remaining: {:0.5e} (train|test) step time: {:0.3f}'.format(
                        epoch + 1,
                        args.epochs, 
                        i + 1, 
                        trainloader['batches'],
                        running_loss, loss.item(),
                        training_cross_entropy_loss.item(), 
                        cross_entropy_loss.item(), 
                        training_accuracy.item(), 
                        test_accuracy.item(), 
                        architecture_reduction.item(), 
                        dtCycle
                    )
                    if args.job is True:
                        print(msg)
                    else:
                        tqdm.write(msg)
                    running_loss = 0.0

                iSave = 100
                if i % iSave == iSave-1:    # print every iSave mini-batches
                    img = plotsearch.plot(cell_weights)
                    if img.size > 0:
                        is_success, buffer = cv2.imencode(".png", img, compression_params)
                        img_enc = io.BytesIO(buffer).read()
                        filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                        s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)
                    imgrad = plotgrads.plot(model)
                    if imgrad.size > 0:
                        is_success, buffer = cv2.imencode(".png", imgrad)  
                        img_enc = io.BytesIO(buffer).read()
                        filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                        s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)
                        # Save calls zero_grads so call it after plotgrads.plot

                    save(model, s3, s3def, args)

                if args.minimum and i+1 >= test_freq:
                    break

                if profile is not None:
                    profile.step()
            #except:
            except NameError:
                print ("Unhandled error in train loop.  Continuing")

            results['batches'] += 1
            if args.minimum and i >= test_freq:
                break

        try:
            #scheduler1.step()
            scheduler2.step()

            img = plotsearch.plot(cell_weights)
            if img.size > 0:
                is_success, buffer = cv2.imencode(".png", img, compression_params)
                img_enc = io.BytesIO(buffer).read()
                filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

            # Plot gradients before saving which clears the gradients
            imgrad = plotgrads.plot(model)
            if imgrad.size > 0:
                is_success, buffer = cv2.imencode(".png", imgrad)  
                img_enc = io.BytesIO(buffer).read()
                filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

            save(model, s3, s3def, args)

            if args.minimum:
                break

            msg = 'epoch {} step {} model {} training complete'.format(epoch, i, args.model_dest)
            if args.job is True:
                print(msg)
            else:
                tqdm.write(msg)
            results['training'] = {}
            if cross_entropy_loss: results['training']['cross_entropy_loss']=cross_entropy_loss.item()
            if architecture_loss: results['training']['architecture_loss']=architecture_loss.item()
            if prune_loss: results['training']['prune_loss']=prune_loss.item()
            if architecture_reduction: results['training']['architecture_reduction']=architecture_reduction.item()
            if training_accuracy: results['training']['accuracy'] =  training_accuracy.item()

            if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.tb_dest is not None and len(args.tb_dest) > 0):
                tb_path = '{}/{}/{}'.format(s3def['sets']['model']['prefix'],args.model_class,args.tb_dest )
                s3.PutDir(s3def['sets']['test']['bucket'], args.tensorboard_dir, tb_path )

        #except:
        except NameError:
            print ("Unhandled error in epoch reporting.  Continuing")
    return results

def Test(args, s3, s3def, model, loaders, device, results, writer, profile=None):
    torch.cuda.empty_cache()
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    test_summary = {'date':date_time}

    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset)) 

    if args.test_dir is not None:
        outputdir = '{}/{}'.format(args.test_dir,args.model_class)
        os.makedirs(outputdir, exist_ok=True)
    else:
        outputdir = None

    accuracy = 0.0
    dtSum = 0.0
    inferTime = []
    for i, data in tqdm(enumerate(testloader['dataloader']), 
                        total=testloader['batches'], 
                        desc="Test steps", 
                        disable=args.job, 
                        bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
        inputs, labels = data

        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        initial = datetime.now()
        with torch.no_grad():
            outputs = model(inputs)

            classifications = torch.argmax(outputs, 1)
        dt = (datetime.now()-initial).total_seconds()
        dtSum += dt
        inferTime.append(dt/args.batch_size)
        tqdm.write('inferTime = {}'.format(inferTime[-1]))
        writer.add_scalar('test/infer', inferTime[-1], results['batches'])
        top1_correct = (classifications == labels).float()
        accuracy += torch.sum(top1_correct).item()

        if args.minimum and i+1 >= 10:
            break

        if profile is not None:
            profile.step()

    test_summary['results'] = {
            'accuracy': args.batch_size*testloader['batches'],
            'minimum time': float(np.min(inferTime)),
            'average time': float(dtSum/testloader['length']),
            'num images': testloader['length'],
        }
    test_summary['object store'] =s3def
    test_summary['config'] = args.__dict__
    if args.ejector is not None and type(args.ejector) != str:
        test_summary['config']['ejector'] = args.ejector.value
    test_summary['system'] = results['system']
    test_summary['training_results'] = results

    # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
    test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
    training_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)
    if training_data is None or type(training_data) is not list:
        training_data = []
    training_data.append(test_summary)
    s3.PutDict(s3def['sets']['test']['bucket'], test_path, training_data)

    test_url = s3.GetUrl(s3def['sets']['test']['bucket'], test_path)
    print("Test results {}".format(test_url))

    results['test'] = test_summary['results']
    return results


def Prune(args, s3, s3def, model, loaders, results):
    initial_parameters = results['initial_parameters']
    model.ApplyStructure()
    reduced_parameters = count_parameters(model)

    results['parameters_after_prune'], results['flops_after_prune'] = ModelSize(args, model, results, loaders)

    save(model, s3, s3def, args)
    results['prune'] = {'final parameters':reduced_parameters, 
                        'initial parameters' : initial_parameters, 
                        'remaining ratio':reduced_parameters/initial_parameters, 
                        'final flops': results['flops_after_prune'], 
                        'initial flops': results['initial_flops'], 
                        'remaining flops':results['flops_after_prune']/results['initial_flops'] }
    print('{} prune results {}'.format(args.model_dest, yaml.dump(results['prune'], default_flow_style=False)))

    return results

def onnx(model, s3, s3def, args, input_channels):
    import torch.onnx as torch_onnx

    dummy_input = torch.randn(args.batch_size, input_channels, args.height, args.width, device='cuda')
    input_names = ["image"]
    output_names = ["segmentation"]
    oudput_dir = '{}/{}'.format(args.test_dir,args.model_class)
    output_filename = '{}/{}.onnx'.format(oudput_dir, args.model_dest)
    dynamic_axes = {input_names[0] : {0 : 'batch_size'},    # variable length axes
                            output_names[0] : {0 : 'batch_size'}}

    os.makedirs(oudput_dir, exist_ok=True)
    torch.onnx.export(model,               # model being run
                dummy_input,                         # model input (or a tuple for multiple inputs)
                output_filename,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = input_names,   # the model's input names
                output_names = output_names, # the model's output names
                dynamic_axes=dynamic_axes,
                opset_version=11)

    succeeded = s3.PutFile(s3def['sets']['model']['bucket'], output_filename, '{}/{}'.format(s3def['sets']['model']['prefix'],args.model_class) )
    if succeeded:
        os.remove(output_filename)

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def main(args):
    print('cell2d test')

    results={}
    results['config'] = args.__dict__
    results['config']['ejector'] = args.ejector.value
    results['system'] = {
        'platform':str(platform.platform()),
        'python':str(platform.python_version()),
        'numpy': str(np.__version__),
        'torch': str(torch.__version__),
        'OpenCV': str(cv2.__version__),
        'pymlutil': str(pymlutil_version.__version__),
        #'torchdatasetutil':str(torchdatasetutil_version.__version__),
    }
    print('cell2d system={}'.format(yaml.dump(results['system'], default_flow_style=False) ))
    print('cell2d config={}'.format(yaml.dump(results['config'], default_flow_style=False) ))

    #torch.autograd.set_detect_anomaly(True)

    s3, _, s3def = Connect(args.credentails)

    results['store'] = s3def

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")

    # Load dataset
    if args.dataset == 'cifar10':
        loaders = CreateCifar10Loaders(args.dataset_path, batch_size = args.batch_size,  
                                       num_workers=args.num_workers, 
                                       cuda = args.cuda, 
                                       rotate=args.augment_rotation, 
                                       scale_min=args.augment_scale_min, 
                                       scale_max=args.augment_scale_max, 
                                       offset=args.augment_translate_x)
        results['classes'] = loaders[0]['classes']
    elif args.dataset == 'imagenet':
        loaders = CreateImagesetLoaders(s3, s3def, 
                                        args.obj_imagenet, 
                                        args.dataset_path+'/imagenet', 
                                        width=args.width, 
                                        height=args.height, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        cuda = args.cuda)


    # Load number of previous batches to continue tensorboard from previous training
    prevresultspath = None
    print('prevresultspath={}'.format(args.prevresultspath))
    if args.prevresultspath and len(args.prevresultspath) > 0:
        prevresults = ReadDict(args.prevresultspath)
        if prevresults is not None:
            if 'batches' in prevresults:
                print('found prevresultspath={}'.format(yaml.dump(prevresults, default_flow_style=False)))
                results['batches'] = prevresults['batches']
            if 'initial_parameters' in prevresults:
                results['initial_parameters'] = prevresults['initial_parameters']
                results['initial_flops'] = prevresults['initial_flops']

    classify, results = load(s3, s3def, args, loaders, results)

    # Prune with loaded parameters than apply current search_structure setting
    classify.ApplyParameters(weight_gain=args.weight_gain, 
                            sigmoid_scale=args.sigmoid_scale,
                            feature_threshold=args.feature_threshold,
                            search_structure=args.search_structure, 
                            convMaskThreshold=args.convMaskThreshold, 
                            k_prune_sigma=args.k_prune_sigma,)


    # # Enable multi-gpu processing
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(classify)
    #     classify = model.module
    # else:
    #     model = classify

    tb = None
    writer = None
    writer_path = '{}/{}'.format(args.tensorboard_dir, args.model_dest)

    # Load previous tensorboard for multi-step training
    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.tb_dest is not None and len(args.tb_dest) > 0):
        tb_path = '{}/{}/{}'.format(s3def['sets']['model']['prefix'],args.model_class,args.tb_dest )
        s3.GetDir(s3def['sets']['test']['bucket'], tb_path, args.tensorboard_dir )
    # Create tensorboard server and tensorboard writer
    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)

        print(f"To launch tensorboard server: tensorboard --bind_all --logdir {args.tensorboard_dir}") # https://stackoverflow.com/questions/47425882/tensorboard-logdir-with-s3-path
        writer = SummaryWriter(writer_path)

    if 'batches' not in results:
        results['batches'] = 0


    if args.prune:
        results = Prune(args, s3, s3def, model=classify, loaders=loaders, results=results)

    if args.train:
        if args.profile:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(writer_path),
                    record_shapes=True, profile_memory=False, with_stack=True, with_flops=False, with_modules=True
            ) as prof:
                results = Train(args, s3, s3def, classify, loaders, device, results, writer, prof)
        else:
            results = Train(args, s3, s3def, classify, loaders, device, results, writer)

    if args.test:
        if args.profile:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=3, repeat=0),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(writer_path),
                    record_shapes=False, profile_memory=False, with_stack=True, with_flops=False, with_modules=True
            ) as prof:
                results = Test(args, s3, s3def, classify, loaders, device, results, writer, prof)
        else:
            results = Test(args, s3, s3def, classify, loaders, device, results, writer)

    if args.onnx:
        onnx(classify, s3, s3def, args, loaders[0]['in_channels'])

    if args.resultspath is not None and len(args.resultspath) > 0:
        WriteDict(results, args.resultspath)

    print('Finished {}'.format(args.model_dest, yaml.dump(results, default_flow_style=False) ))
    print(json.dumps(results, indent=2))
    return 0

if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach on {}:{}".format(args.debug_address, args.debug_port))
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

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = main(args)
    sys.exit(result)

