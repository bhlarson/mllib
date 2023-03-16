#!/usr/bin/python3
import math
import os
import sys
import io
import json
import yaml
import platform
import re
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
import torchvision
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import cv2

from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDict, WriteDict, str2bool
from pymlutil.s3 import s3store, Connect
from pymlutil.functions import Exponential, GaussianBasis
from pymlutil.metrics import DatasetResults
import pymlutil.version as pymlutil_version
from pymlutil.version import VersionString

from torchdatasetutil.imagenetstore import CreateImagenetLoaders
from torchdatasetutil.cifar10store import CreateCifar10Loaders
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, os.path.abspath(''))
import train.presets as presets
import train.transforms as transforms
import train.utils as utils
from train.sampler import RASampler

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def prep_img(img):
    img = img.cpu().numpy()
    if img.shape[1] == 1 or len(img.shape) < 3:
        one_channel = True
    else:
        one_channel = False
    if one_channel:
        img = img.mean()
    if img.max()-img.min() > 0:
        scale = img.max()-img.min()
    else:
        scale = 1.0
    img = 255* (img - img.min())/scale # unnormalize
    img = img.astype('uint8')
    if one_channel:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    else:
        img = np.transpose(img, (1, 2, 0))

    return img

def plot_classes_preds(images, labels, preds, loader, imsize = 12, numimages = 4):
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(imsize, imsize*numimages))
    classifications = torch.argmax(preds, 1)
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(classifications, preds)]

    for idx in np.arange(numimages):
        ax = fig.add_subplot(1, numimages, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(plt, images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            loader['classes'][classifications[idx]],
            probs[idx] * 100.0,
            loader['classes'][labels[idx].item()]),
            color=("green" if classifications[idx]==labels[idx].item() else "red"))
    return fig


def PlotClassPredictions(images, labels, preds, loader, numimages = 4):

    classifications = torch.argmax(preds, 1)
    probabilities = [F.softmax(el, dim=0)[i].item() for i, el in zip(classifications, preds)]

    #images = np.squeeze(images)
    #labels = np.squeeze(labels)
    if len(images) < numimages:
        numimages = len(images)

    plot_images = []
    for idx in np.arange(numimages):
        img = prep_img(images[idx])
        classifications_txt = "{0} {1:.1f}%".format(
            loader['classes'][classifications[idx]],
            probabilities[idx] * 100.0,
            loader['classes'][labels[idx].item()])
        label_text = "{}".format(loader['classes'][labels[idx].item()])
        color=((0,255,0) if classifications[idx]==labels[idx].item() else (255,0,0))
        fontScale = 1
        img = cv2.putText(img.astype(np.uint8).copy(), classifications_txt,(1,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale,color,1,cv2.LINE_AA)
        img = cv2.putText(img.astype(np.uint8).copy(), label_text,(1,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale,color,1,cv2.LINE_AA)
        plot_images.append(img)

    plot_images = cv2.hconcat(plot_images)
    #plot_images = cv2.cvtColor(plot_images, cv2.COLOR_BGR2RGB)

    return plot_images



sys.path.insert(0, os.path.abspath(''))
from networks.totalloss import TotalLoss, FenceSitterEjectors

# Inner neural architecture cell repetition structure
# Process: Con2d, optional batch norm, optional ReLu

def relu_flops_counter_hook(module, shape, relaxation):
    flops_total = np.prod(shape)
    flops_relaxed = flops_total

    if relaxation is not None:
        # Evaluate new way to compute flops_relaxed
        # shape_relaxed = shape
        # shape_relaxed[1] = torch.sum(relaxation.weights())
        # flops_relaxed = np.prod(shape_relaxed)

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
            if in_relaxation_source is not None:
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

def bn_params_counter_hook(module, shape, relaxation):
    params_total = model_weights(module)

    if relaxation is not None:
        params_relaxed = torch.sum(relaxation.weights())
    else:
        params_relaxed = params_total

    return params_relaxed, params_total

def conv_params_counter_hook(conv_module, in_relaxation, output_shape, out_relaxation):
    # Can have multiple inputs, getting the first one
    output_dims = list(output_shape)[2:]

    kernel_dims = list(conv_module.kernel_size)
    #in_channels = conv_module.in_channels
    input_relaxation = []
    if in_relaxation is not None and len(in_relaxation) > 0:
        for in_relaxation_source in in_relaxation:
            if in_relaxation_source is not None:
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

    if conv_module.bias is not None:
        bias_flops = out_channel_weight * out_channels
    else:
        bias_flops = 0

    conv_base = int(np.prod(kernel_dims)) / groups
    conv_params_relaxed = conv_base * in_channel_weight * out_channel_weight + bias_flops
    conv_params_total = conv_base * in_channels * out_channels + bias_flops

    return conv_params_relaxed, conv_params_total

class RelaxChannels(nn.Module):
    def __init__(self,channels, 
                 device=torch.device("cpu"),
                 search_structure=True,
                 disable_search_structure = False,
                 sigmoid_scale = 5.0, # Channel sigmoid scale factor
                 init_mean = 0.0, # 
                 init_std = 0.01, # Variable initiation standard deviation
                 prevent_collapse=False,
                ):
        super(RelaxChannels, self).__init__()
        self.channels = channels
        self.device = device
        self.search_structure = search_structure
        self.disable_search_structure = disable_search_structure
        self.sigmoid_scale = sigmoid_scale
        self.channel_scale = nn.Parameter(torch.zeros(self.channels, dtype=torch.float, device=self.device))
        self.init_mean = init_mean
        self.init_std = init_std
        self.prevent_collapse = prevent_collapse

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.ones_(self.channel_scale)
        #nn.init.normal_(self.channel_scale, mean=self.init_mean, std = self.init_std)

    def ApplyParameters(self, search_structure=None, sigmoid_scale=None):
        if search_structure is not None:
            self.disable_search_structure = not search_structure
        if sigmoid_scale is not None:
            self.sigmoid_scale = sigmoid_scale

    def scale(self):
        return self.channel_scale

    def weights(self):
        conv_weights = self.sigmoid(self.sigmoid_scale*self.channel_scale)

        if 'prevent_collapse' in self.__dict__ and self.prevent_collapse is not None and self.prevent_collapse:
            #conv_weights = conv_weights/conv_weights.max().item()
            conv_weights = conv_weights/conv_weights.max()
        
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
        self.channels = len(self.channel_scale.data)

        print("{} channels={}".format(prefix, self.channels))

        return conv_mask

class ConvBR(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 prev_relaxation = None,
                 relaxation = None,
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
                 max_pool = False,
                 pool_kernel_size=3,
                 pool_stride=2,
                 pool_padding=1,
                 prevent_collapse=False
                 ):
        super(ConvBR, self).__init__()
        self.in_channels = in_channels
        if out_channels < 1:
            raise ValueError("out_channels must be > 0")
        self.out_channels = out_channels
        self.prev_relaxation = prev_relaxation
        self.relaxation = relaxation
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
        self.max_pool = max_pool
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.prevent_collapse = prevent_collapse


        if prev_relaxation is not None:
            assert type(prev_relaxation)== list

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

        if self.relaxation is None:
            if self.search_structure:
                self.relaxation = RelaxChannels(self.out_channels, 
                                                device=self.device, 
                                                search_structure=self.search_structure,
                                                disable_search_structure=self.disable_search_structure,
                                                sigmoid_scale=self.sigmoid_scale,
                                                prevent_collapse=self.prevent_collapse,
                                                )
            else:
                self.relaxation = None

        if self.max_pool:
                self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride, padding=self.pool_padding)
        else:
            self.pool = None

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
                        sigmoid_scale=None, weight_gain=None, k_prune_sigma=None, search_flops=None, batch_norm=None):
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
        if batch_norm is not None:
            self.batch_norm = batch_norm

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

            if hasattr(self, 'pool') and self.pool is not None:
                x = self.pool(x)

        else :
            print("Failed to prune zero size convolution")

        if self.output_shape is None:
            self.output_shape = x.shape

        return x

    def ArchitectureWeights(self):

        if self.relaxation and self.search_structure and not self.disable_search_structure:
            weight_basis = GaussianBasis(self.relaxation.scale(), sigma=self.k_prune_sigma)
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
                architecture_weights, cell_weights = conv_params_counter_hook(self.conv, self.prev_relaxation, self.output_shape, self.relaxation)

                if self.batch_norm:
                    params_relaxed, params_total = bn_params_counter_hook(self.batchnorm2d, self.output_shape, self.relaxation)
                    architecture_weights += params_relaxed
                    cell_weights += params_total

                if not torch.is_tensor(architecture_weights):
                    architecture_weights = torch.tensor(architecture_weights, device = self.device)

                architecture_weights = torch.reshape(architecture_weights,[-1]) # reshape to single element array to be the same format as not flops architecture_weights
                
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
            conv_weights = self.relaxation.weights()
            conv_mask = conv_weights > self.convMaskThreshold
        else:
            conv_mask = torch.ones((self.out_channels), dtype=torch.bool, device=self.device) # Always true if self.search_structure == False

        shared_relaxation = False
        if out_channel_mask is not None:
            if len(out_channel_mask) == len(conv_mask):
                # If either mask is false, then the convolution is removed (not nand)
                conv_mask = torch.logical_not(torch.logical_or(torch.logical_not(conv_mask), torch.logical_not(out_channel_mask)))     
            else:
                conv_mask = out_channel_mask # Shared relaxation weights
                shared_relaxation = True
                

        prev_convolutions = len(conv_mask)
        pruned_convolutions = len(conv_mask[conv_mask==False])

        if self.residual and prev_convolutions==pruned_convolutions: # Pruning full convolution 
            conv_mask = ~conv_mask # Residual connection becomes a straight through connection
            self.out_channels = 0
        else:
            self.out_channels = len(conv_mask[conv_mask!=0])
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias[conv_mask!=0]
            if self.conv_transpose:
                self.conv.weight.data = self.conv.weight[:, conv_mask!=0]
            else:
                self.conv.weight.data = self.conv.weight[conv_mask!=0]
            self.conv.in_channels = self.in_channels
            self.conv.out_channels = self.out_channels

            #self.channel_scale.data = self.channel_scale.data[conv_mask!=0]

            if self.batch_norm:
                self.batchnorm2d.bias.data = self.batchnorm2d.bias.data[conv_mask!=0]
                self.batchnorm2d.weight.data = self.batchnorm2d.weight.data[conv_mask!=0]
                self.batchnorm2d.running_mean = self.batchnorm2d.running_mean[conv_mask!=0]
                self.batchnorm2d.running_var = self.batchnorm2d.running_var[conv_mask!=0]
                self.batchnorm2d.num_features = self.out_channels

            if not shared_relaxation and self.relaxation and self.search_structure and not self.disable_search_structure:
                self.relaxation.ApplyStructure(conv_mask)

        print("{} {}={}/{} in_channels={} out_channels={}".format(prefix, pruned_convolutions/prev_convolutions, pruned_convolutions, prev_convolutions, self.in_channels, self.out_channels))

        return conv_mask


DefaultMaxDepth = 1
class Cell(nn.Module):
    def __init__(self,
                 in1_channels, 
                 in2_channels = 0,
                 prev_relaxation = None,
                 relaxation = None,
                 residual_relaxation = None,
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
                 prevent_collapse = None
                 ):
                
        super(Cell, self).__init__()

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.prev_relaxation = prev_relaxation
        self.relaxation = relaxation
        self.residual_relaxation = residual_relaxation
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
        self.prevent_collapse = prevent_collapse


        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convolutions uses out_channels as chanels

        src_channels = in_chanels = self.in1_channels+self.in2_channels

        totalStride = 1
        totalDilation = 1
        prev_relaxation = self.prev_relaxation
        relaxation = self.relaxation
        self.activation = None

        if self.residual and (in_chanels != self.convolutions[-1]['out_channels'] or totalStride != 1 or totalDilation != 1):
            for i, convdev in enumerate(convolutions):
                totalStride *= convdev['stride']
                totalDilation *= convdev['dilation']

            self.conv_residual = ConvBR(in_chanels, self.convolutions[-1]['out_channels'],
                prev_relaxation = prev_relaxation,
                relaxation = relaxation,
                batch_norm=self.batch_norm, 
                relu=False, 
                kernel_size=1, 
                stride=totalStride, 
                dilation=totalDilation, 
                groups=self.groups, 
                bias=self.bias, 
                padding_mode=self.padding_mode,
                weight_gain=self.weight_gain,
                convMaskThreshold=self.convMaskThreshold, 
                dropout_rate=self.dropout_rate,
                search_structure=self.search_structure,
                residual=True,
                dropout=self.dropout,
                k_prune_sigma=self.k_prune_sigma,
                device=self.device,
                prevent_collapse = True)

            self.residual_relaxation = self.conv_residual.relaxation

        else:
            self.conv_residual = None

        if self.residual and self.relu:
            self.activation = nn.ReLU()

        for i, convdev in enumerate(convolutions):
            conv_transpose = False
            if 'conv_transpose' in convdev and convdev['conv_transpose']:
                conv_transpose = True

            relu = self.relu
            if 'relu' in convdev:
                relu = convdev['relu']

            batch_norm = self.batch_norm
            if 'batch_norm' in convdev:
                batch_norm = convdev['batch_norm']

            max_pool = False
            if 'max_pool' in convdev:
                max_pool = convdev['max_pool']

            pool_kernel_size = 3
            if 'pool_kernel_size' in convdev:
                pool_kernel_size = convdev['pool_kernel_size']

            pool_stride = 2
            if 'pool_stride' in convdev:
                pool_stride = convdev['pool_stride']

            pool_padding = 1
            if 'pool_padding' in convdev:
                pool_padding = convdev['pool_padding']

            prevent_collapse = False
            if self.prevent_collapse is not None:
                prevent_collapse = self.prevent_collapse
            else:
                if 'prevent_collapse' in convdev:
                    prevent_collapse = convdev['prevent_collapse']
            if convdev['stride'] > 1:
                prevent_collapse = True

            relu = relu
            if 'relu' in convdev:
                relu = convdev['relu']

            # Apply the residual relaxation to the output convolution's relaxation
            if i == len(convolutions)-1:
                relaxation = self.residual_relaxation

            conv = ConvBR(src_channels, convdev['out_channels'], 
                prev_relaxation = prev_relaxation,
                relaxation = relaxation,
                batch_norm=batch_norm, 
                relu=relu, 
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
                search_flops = self.search_flops,
                max_pool = max_pool,
                pool_kernel_size = pool_kernel_size,
                pool_padding = pool_padding,
                prevent_collapse = prevent_collapse,
                )

            prev_relaxation = [conv.relaxation]
            relaxation = None
            
            self.cnn.append(conv)

            src_channels = convdev['out_channels']

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
                        k_prune_sigma=None, search_flops=None, batch_norm=None): # Apply a parameter change
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

        if batch_norm is not None:
            self.batch_norm = batch_norm

        if self.cnn is not None and len(self.cnn) > 0:
            if self.conv_residual is not None:
                self.conv_residual.ApplyParameters(search_structure=search_structure, convMaskThreshold=convMaskThreshold, dropout=dropout,
                                     weight_gain=weight_gain, sigmoid_scale=sigmoid_scale, k_prune_sigma=k_prune_sigma, search_flops=search_flops, batch_norm=batch_norm)


            for conv in self.cnn:
                conv.ApplyParameters(search_structure=search_structure, convMaskThreshold=convMaskThreshold, dropout=dropout,
                                     weight_gain=weight_gain, sigmoid_scale=sigmoid_scale, k_prune_sigma=k_prune_sigma, search_flops=search_flops, batch_norm=batch_norm)

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
        
        if self.residual:
            if self.conv_residual is not None:
                layermsg = "cell residual search_structure={}".format(self.conv_residual.search_structure)
                if msg is not None:
                    layermsg = "{} {} ".format(msg, layermsg)
                
                residual_out_channel_mask = self.conv_residual.ApplyStructure(in_channel_mask=in_channel_mask, msg=layermsg)
                residual_out_channels = self.conv_residual.out_channels
            else:
                residual_out_channel_mask = in_channel_mask

        if self.cnn is not None:

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

                    if self.residual and i == len(self.cnn)-1:
                        out_channel_mask = residual_out_channel_mask
                    else:
                        out_channel_mask = None
                        
                    out_channel_mask = cnn.ApplyStructure(in_channel_mask=in_channel_mask, out_channel_mask = out_channel_mask, msg=layermsg)
                    out_channels = cnn.out_channels
                    if out_channels == 0: # Prune convolutions if any convolution has no more outputs
                        if i != len(self.cnn)-1: # Make mask the size of the cell output with all values 0
                            out_channel_mask = torch.zeros(self.cnn[-1].out_channels, dtype=np.bool, device=self.device)

                        layermsg = "Prune cell because convolution {} out_channels == {}".format(i, out_channels)
                        if msg is not None:
                            layermsg = "{} {} ".format(msg, layermsg)
                        print(layermsg)

                        self.cnn = None
                        if self.residual:
                            out_channel_mask = residual_out_channel_mask
                            out_channels = len(residual_out_channel_mask[residual_out_channel_mask!=0])
                        else:
                            out_channels = 0
                        break
                    else:
                        in_channel_mask = out_channel_mask

        else:
            if self.residual:
                out_channel_mask = residual_out_channel_mask
                out_channels = len(residual_out_channel_mask[residual_out_channel_mask!=0])
            else:
                out_channel_mask = None
                out_channels = 0


        layermsg = "cell summary: weights={} in1_channels={} in2_channels={} out_channels={} residual={} search_structure={} passthrough={}".format(
            self.total_trainable_weights, 
            self.in1_channels, 
            self.in2_channels, 
            out_channels,
            self.residual,
            self.search_structure,
            self.cnn == None)
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

        if self.activation:
            y = self.activation(y)

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


def ResnetCells(size = Resnet.layers_50, useConv1 = True, conv1_kernel_size = 7, conv1_stride = 2):
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
        cell.append({'out_channels':network_channels[0], 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True})
        resnetCells.append({'residual':False, 'cell':cell})
    else:
        network_channels = [64, 128, 256, 512]
        if useConv1:
            cell.append({'out_channels':network_channels[0], 'kernel_size': conv1_kernel_size, 'stride': conv1_stride, 'dilation': 1, 'search_structure':True, 'residual':False, 'max_pool': True})
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
                cell.append({'out_channels':network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':True, 'relu': True})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1, 'search_structure':True, 'relu': True})
                cell.append({'out_channels':4*network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':True, 'relu': False})
            else:
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1, 'search_structure':True, 'relu': True})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True, 'relu': False})
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

        prev_relaxation = None
        residual_relaxation = None
        for i, cell_convolutions in enumerate(convolutions):

            convdfn = None

            if cell_convolutions['residual'] and residual_relaxation is None:
                residual_relaxation = prev_relaxation

            if i == 0:
                prevent_collapse = True
            else:
                prevent_collapse = False

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
                search_flops = self.search_flops,
                prevent_collapse = prevent_collapse,
                prev_relaxation = [prev_relaxation],
                residual_relaxation = residual_relaxation,
                bias = False
                )

            if cell.conv_residual is not None:
                prev_relaxation = cell.conv_residual.relaxation
                residual_relaxation = cell.conv_residual.relaxation
            elif cell.cnn[-1].relaxation is not None:
                prev_relaxation = cell.cnn[-1].relaxation
            in_channels = cell_convolutions['cell'][-1]['out_channels']
            self.cells.append(cell)

        #self.maxpool = nn.MaxPool2d(2, 2) # Match Pytorch pretrained Resnet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FC(in_channels, self.out_channels, device=self.device)

        self.total_trainable_weights = model_weights(self)

        self.fc_weights = model_weights(self.fc)

    def ApplyStructure(self):
        layermsg = 'Initial resize convolution'
        #in_channel_mask = self.resnet_conv1.ApplyStructure(msg=layermsg)
        in_channel_mask = None
        for i, cell in enumerate(self.cells):
            layermsg = 'Cell {}'.format(i)
            out_channel_mask = cell.ApplyStructure(in1_channel_mask=in_channel_mask, msg=layermsg)
            in_channel_mask = out_channel_mask

        self.fc.ApplyStructure(in_channel_mask=in_channel_mask)

    def ApplyParameters(self, search_structure=None, convMaskThreshold=None, dropout=None, 
                        weight_gain=None, sigmoid_scale=None, feature_threshold=None,
                        k_prune_sigma=None, search_flops=None, batch_norm=None): # Apply a parameter change
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
        if batch_norm is not None:
            self.batch_norm = batch_norm

        for cell in self.cells:
            cell.ApplyParameters(search_structure=search_structure, dropout=dropout, convMaskThreshold=convMaskThreshold,
                                 weight_gain=weight_gain, sigmoid_scale=sigmoid_scale, feature_threshold=feature_threshold,
                                 k_prune_sigma=k_prune_sigma, search_flops=search_flops, batch_norm=batch_norm)

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

        for i,  in_cell in enumerate(self.cells):
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
    parser.add_argument('-s3_name', type=str, default='store', help='S3 name in credentials')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152, 20, 32, 44, 56, 110], default=56, help='Run description')
    parser.add_argument('-useConv1', type=str2bool, default=False, help='If true, use initial convolution and max pool before ResNet blocks')

    parser.add_argument('-dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'], help='Dataset')
    parser.add_argument('-dataset_path', type=str, default='/data', help='Local dataset path')
    parser.add_argument('-obj_imagenet', type=str, default='data/imagenet', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-batch_size', type=int, default=1000, help='Training batch size') 

    parser.add_argument('-optimizer', type=str, default='adam', choices=['sgd', 'rmsprop', 'adam', 'adamw'], help='Optimizer')
    parser.add_argument('-learning_rate', type=float, default=2e-5, help='Training learning rate')
    parser.add_argument('-learning_rate_decay', type=float, default=1e-4, help='Rate decay multiple')
    parser.add_argument('-rate_schedule', type=json.loads, default='[8, 12, 15, 18]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[40, 60, 65]', help='Training learning rate')
    #parser.add_argument('-rate_schedule', type=json.loads, default='[10, 15, 17]', help='Training learning rate')
    
    parser.add_argument('-momentum', type=float, default=0.9, help='Learning Momentum')
    parser.add_argument('-weight_decay', type=float, default=1e-4)
    parser.add_argument('-epochs', type=int, default=220, help='Training epochs')
    parser.add_argument('-start_epoch', type=int, default=0, help='Start epoch')

    parser.add_argument('-num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('-model_type', type=str,  default='classification')
    parser.add_argument('-model_class', type=str,  default='ImgClassifyPrune')
    parser.add_argument('-model_src', type=str,  default="20230308_055658_ipc_search_structure_06")
    parser.add_argument('-model_dest', type=str, default="20230303_204145_ipc_search_structure_02")
    parser.add_argument('-test_sparsity', type=int, default=10, help='test step multiple')
    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-cuda', type=bool, default=True)

    parser.add_argument('-height', type=int, default=32, help='Input image height')
    parser.add_argument('-width', type=int, default=32, help='Input image width')
    parser.add_argument('-channels', type=int, default=3, help='Input image color channels')
    parser.add_argument('-k_accuracy', type=float, default=1.0, help='Accuracy weighting factor')
    parser.add_argument('-k_structure', type=float, default=2.0, help='Structure minimization weighting factor')
    parser.add_argument('-k_prune_basis', type=float, default=1.0, help='prune base loss scaling')
    parser.add_argument('-k_prune_exp', type=float, default=50.0, help='prune basis exponential weighting factor')
    parser.add_argument('-k_prune_sigma', type=float, default=1.0, help='prune basis exponential weighting factor')
    parser.add_argument('-target_structure', type=float, default=0.0, help='Structure minimization weighting factor')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.5, help='tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.025, help='convolution channel sigmoid level to prune convolution channels')

    parser.add_argument('-augment_rotation', type=float, default=0.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_min', type=float, default=1.0, help='Input augmentation scale')
    parser.add_argument('-augment_scale_max', type=float, default=1.0, help='Input augmentation scale')
    parser.add_argument('-augment_translate_x', type=float, default=0.0, help='Input augmentation translation')
    parser.add_argument('-augment_translate_y', type=float, default=0.0, help='Input augmentation translation')
    parser.add_argument('-augment_noise', type=float, default=0.0, help='Augment image noise')

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
    parser.add_argument('-onnx', type=str2bool, default=False)
    parser.add_argument('-write_vision_graph', type=str2bool, default=False)
    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-test_name', type=str, default='default_test', help='Test name for test log' )
    parser.add_argument('-test_path', type=str, default='test/tests.yaml', help='S3 path to test log')
    parser.add_argument('-resultspath', type=str, default='results.yaml')
    parser.add_argument('-prevresultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='/tb_logs/20230303_204145_ipc_tb', help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    #parser.add_argument('-tensorboard_dir', type=str, default=None, help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    parser.add_argument('-tb_dest', type=str, default='20230303_204145_ipc_tb')
    parser.add_argument('-config', type=str, default='config/build.yaml', help='Configuration file')
    parser.add_argument('-description', type=str, default='Resnet classification', help='Test description')
    parser.add_argument('-output_dir', type=str, default='./out', help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')


    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", dest="lr_warmup_epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", dest="lr_warmup_method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=4, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )


    args = parser.parse_args()

    if args.d:
        args.debug = args.d
    if args.min:
        args.minimum = args.min

    return args

def ModelSize(args, model, loaders):

    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset))

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")

    input = torch.zeros((1, testloader['in_channels'], args.height, args.width), device=device)
    model(input)

    image_flops, parameters = get_model_complexity_info(deepcopy(model), (testloader['in_channels'], args.height, args.width), as_strings=False,
                                         print_per_layer_stat=False, verbose=False)

    # flops = FlopCountAnalysis(model, input)
    # parameters = count_parameters(model)
    #image_flops = flops.total()

    return parameters, image_flops


def WriteModelGraph(args, writer, model, loaders):

    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset))

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")

    input = torch.zeros((1, testloader['in_channels'], args.height, args.width), device=device)
    writer.add_graph(model, input)

def InitWeights(model_state_dict, tv_state_dict, useConv1 = True):
    mkeys = list(model_state_dict.keys())
    tvkeys = tv_state_dict.keys()

    if useConv1:
        iCell = 0
    else:
        iCell = -1
    imkeys = 0

    iConv = 0
    pBlock = -1
    pResidual = -1
    pLayer = -1
    for i, tvkey in enumerate(tvkeys):
        tvkeyname = re.split('(\d*)\.', tvkey)
        tvkeyname = list(filter(None, tvkeyname))
        mkey = None

        if len(tvkeyname) == 6:
            layer = tvkeyname[1]
            residual = tvkeyname[2]
            block = int(tvkeyname[4])

            if block != pBlock:
                iConv += 1
            
            if residual != pResidual:
                iCell += 1
                iConv = 0

            # if layer != pLayer:
            #     print('layer {}'.format(layer))

            pBlock = block
            pResidual = residual
            pLayer = layer

            blockname = 'cnn'
            if tvkeyname[3] == 'downsample':
                blockname = 'conv_residual'
                if tvkeyname[4] == '0':
                    mkey_operator = 'conv'
                else:
                    mkey_operator = 'batchnorm2d'

                mkey = 'cells.{}.{}.{}.{}'.format(iCell, blockname, mkey_operator,tvkeyname[-1]) 
            else:
                if tvkeyname[3] == 'bn':
                    mkey_operator = 'batchnorm2d'
                else:
                    mkey_operator = tvkeyname[3]

                mkey = 'cells.{}.{}.{}.{}.{}'.format(iCell, blockname, iConv,mkey_operator,tvkeyname[-1]) 

        elif tvkeyname[0] == 'conv' and tvkeyname[1]=='1':
            if useConv1:
                mkey = 'cells.{}.cnn.{}.{}.{}'.format(iCell, iConv,tvkeyname[0],tvkeyname[-1])

        elif tvkeyname[0] == 'bn' and tvkeyname[1]=='1':
            if useConv1:
                mkey = 'cells.{}.cnn.{}.batchnorm2d.{}'.format(iCell, iConv, tvkeyname[-1]) 

        elif tvkeyname[0] == 'fc':
            mkey = 'fc.{}.{}'.format(tvkeyname[0], tvkeyname[1])

        else:
            print('{}: {} skipping'.format(i, tvkey))

        if mkey is not None:
            if mkey in model_state_dict:
                if model_state_dict[mkey].shape == tv_state_dict[tvkey].shape:
                    # print('{}: {} = {}'.format(i, mkey, tvkey))
                    model_state_dict[mkey] = tv_state_dict[tvkey].data.clone()
                else:
                    print('{}: {}={} != {}={} not in model'.format(i, mkey, model_state_dict[mkey].shape, tvkey, tv_state_dict[tvkey].shape))
            else:
                print('{}: {} == {} not in model'.format(i, mkey, tvkey))
        
    return model_state_dict


def load(s3, s3def, args, loaders, results):

    model = MakeNetwork(args, source_channels = loaders[0]['in_channels'], out_channels = loaders[0]['num_classes'])
    model_weights = model.state_dict()

    tv_weights = None
    transforms_vision = None
    model_vision = None
    if args.resnet_len == 18:
        tv_weights = torchvision.models.ResNet18_Weights(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        #model_vision = torchvision.models.resnet18(weights=tv_weights)

    elif args.resnet_len == 34:
        tv_weights = torchvision.models.ResNet34_Weights(torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        #model_vision = torchvision.models.resnet34(weights=tv_weights)

    elif args.resnet_len == 50:
        tv_weights = torchvision.models.ResNet50_Weights(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        #model_vision = torchvision.models.resnet50(weights=tv_weights)

    elif args.resnet_len == 101:
        tv_weights = torchvision.models.ResNet101_Weights(torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        #model_vision = torchvision.models.resnet101(weights=tv_weights)

    elif args.resnet_len == 152:
        tv_weights = torchvision.models.ResNet152_Weights(torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
        #model_vision = torchvision.models.resnet152(weights=tv_weights)

    if tv_weights is not None:
        state_dict = tv_weights.get_state_dict(progress=True)
        model_weights = InitWeights(model_weights, state_dict, useConv1 = args.useConv1)
        model.load_state_dict(state_dict = model_weights, strict = False)
        transforms_vision = tv_weights.transforms()

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")
    model.to(device)
    #model_vision.to(device)

    results['initial_parameters'] , results['initial_flops'] = ModelSize(args, model, loaders)
    print('load initial_parameters = {} initial_flops = {}'.format(results['initial_parameters'], results['initial_flops']))


    if(args.model_src and args.model_src != ''):
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            model = torch.load(io.BytesIO(modelObj))
            
            model_parameters, model_flops = ModelSize(args, model, loaders)
            results['load'][args.model_dest]= {'model_parameters':model_parameters, 'model_flops':model_flops}
            print('load model_parameters = {} model_flops = {}'.format(model_parameters, model_flops))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(
                s3def['sets']['model']['bucket'],
                s3def['sets']['model']['prefix'],
                args.model_class,args.model_src))
            return model

    return model, results, model_vision, transforms_vision

def save(model, s3, s3def, args, loc=''):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    #torch.save(model.state_dict(), out_buffer) # To save just state dictionary, need to determine pruned network from state dict
    torch.save(model, out_buffer)
    outname = '{}/{}/{}{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest,loc)

    print('save {}/{}'.format(s3def['sets']['model']['bucket'], outname))
    succeeded = s3.PutObject(s3def['sets']['model']['bucket'], outname, out_buffer)
    print('s3.PutObject return {}'.format(succeeded))

def save_file(model,outname):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model, outname)

def MakeNetwork(args, source_channels = 3, out_channels = 10):
    resnetCells = ResnetCells(Resnet(args.resnet_len), useConv1 = args.useConv1)

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

    return network

def activation_hook(name, results_dict):
    def hook(model, input, output):
        results_dict[name] = output.detach()
    return hook

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
        if len(weights) > 0:
            for i,  cell, in enumerate(weights):
                if len(cell['cell_weight']) > 0:
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

        conv_mag = '{:0.3e}'.format(max_weight, min_weight)
        cv2.putText(img,conv_mag,(int(0.05*width), int(0.90*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.75, color=(0, 255, 255))

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

class PlotConvMag():
    def __init__(self, title = 'Convolution Magnitude', colormapname = 'jet', lenght = 5, dpi=1200, thickness=1, classification=True, pruning=True ):

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

        conv_mag = []
        max_gradient =  float('-inf')
        min_gradient = float('inf')
        max_layers = 0
        for i,  cell, in enumerate(network.cells):
            if cell.cnn is not None:
                for j, convbr in enumerate(cell.cnn):
                    layer_mags = []
                    if convbr.conv.weight is not None:
                        if convbr.conv_transpose:
                            mags = convbr.conv.weight.permute(1, 0, 2, 3).flatten(1).norm(dim=1)/np.sqrt(np.product(convbr.conv.weight.shape[1:]))
                        else:
                            mags = convbr.conv.weight.flatten(1).norm(dim=1)/np.sqrt(np.product(convbr.conv.weight.shape[0:]))              


                        #x = i*self.lenght*len(cell.cnn)+j*self.lenght

                        for k, mag in enumerate(mags.cpu().detach().numpy()):
                            if mag > max_gradient:
                                max_gradient = mag
                            if mag < min_gradient:
                                min_gradient = mag

                            layer_mags.append(mag)
                            
                            '''y = int(k*self.thickness+self.thickness/2)
                            start_point = (x,y)
                            end_point=(x+self.lenght,y)

                            color = 255*np.array(self.cm(mag))
                            color = color.astype('uint8')
                            colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                            cv2.line(img,start_point,end_point,colorbgr,self.thickness)'''
                    conv_mag.append(layer_mags)
                    max_layers = max(max_layers, len(layer_mags))

        width = len(conv_mag)*self.lenght
        height = max_layers*self.thickness
        img = np.zeros([height,width,3]).astype(np.uint8)

        x = 0
        for j, gradient_norm in enumerate(conv_mag):
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


def LogTest(args, s3, s3def, results):

    min_results = {}
    if 'initial_parameters' in results:
        min_results['initial_parameters'] = results['initial_parameters']
    if 'initial_flops' in results:
        min_results['initial_flops'] = results['initial_flops']
    if 'load' in results:
        min_results['load'] = results['load'] 
    if 'training' in results:
        min_results['training'] = results['training']
    if 'test' in results:
        min_results['test'] = results['test']
    if 'prune' in results:
        min_results['prune'] = results['prune']

    test_data = s3.GetDict(s3def['sets']['test']['bucket'], args.test_path)
    if test_data is None or type(test_data) is not list:
        test_data = []

    iTest = next((idx for idx, test in enumerate(test_data) if 'name' in test and test['name'] == args.test_name), None)
    if iTest is not None:
        test_data[iTest]['results'] = min_results
    else:
        test_time = datetime.now()
        test = {
            'name': args.test_name,
            'when': test_time.strftime("%c"),
            'server': None,
            'image': None,
            'workflow': None,
            'job': None,
            'tensorboard': args.tb_dest,
            'description': args.description,
            'results': min_results,
        }
        test_data.append(test)

    s3.PutDict(s3def['sets']['test']['bucket'], args.test_path, test_data)


default_loaders = [{'set':'train', 'enable_transform':True},
                   {'set':'test', 'enable_transform':False}]

def Train(args, s3, s3def, model, loaders, device, results, writer, profile=None):
    torch.cuda.empty_cache()
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

    opt_name = args.optimizer.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay )

    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    #scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.rate_schedule, gamma=args.learning_rate_decay, verbose=True)

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    plotsearch = PlotSearch()
    plotgrads = PlotGradients()
    plotconvmag = PlotConvMag()


    test_freq = args.test_sparsity*int(math.ceil(trainloader['batches']/testloader['batches']))
    tstart = None
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    # Train
    results['train'][args.model_dest] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'architecture_reduction':[]}
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

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs, isTraining=True)
                classifications = torch.argmax(outputs, 1)
                tinfer = time.perf_counter()
                loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale = loss_fcn(outputs, labels, model)
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
                        img = PlotClassPredictions(inputs, labels, outputs, trainloader)
                        writer.add_image('predictions/train', img, 0, dataformats='HWC')

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

                        img = PlotClassPredictions(inputs, labels, outputs, testloader)
                        writer.add_image('predictions/test', img, results['batches'], dataformats='HWC')

                        imgrad = plotgrads.plot(model)
                        if imgrad.size > 0:
                            im_grad_norm = cv2.cvtColor(imgrad, cv2.COLOR_BGR2RGB)
                            writer.add_image('network/gradient_norm', im_grad_norm, results['batches'],dataformats='HWC')

                        convmag = plotconvmag.plot(model)
                        if convmag.size > 0:
                            convmag = cv2.cvtColor(convmag, cv2.COLOR_BGR2RGB)
                            writer.add_image('network/conv_mag', convmag, results['batches'],dataformats='HWC')


                        if args.search_structure:
                            if cell_weights is not None:
                                imprune_weights = plotsearch.plot(cell_weights)
                                if imprune_weights.size > 0:
                                    im_class_weights = cv2.cvtColor(imprune_weights, cv2.COLOR_BGR2RGB)
                                    writer.add_image('network/prune_weights', im_class_weights, results['batches'], dataformats='HWC')

                    running_loss /=test_freq
                    msg = '[{:3}/{}, {:6d}/{}]  loss: {:0.5e}|{:0.5e} cross-entropy loss: {:0.5e}|{:0.5e} accuracy: {:0.5e}|{:0.5e} remaining: {:0.5e} (train|test) compute time: {:0.3f} cycle time: {:0.3f}'.format(
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
                        dtCompute,
                        dtCycle
                    )
                    if args.job is True:
                        print(msg)
                    else:
                        tqdm.write(msg)
                    running_loss = 0.0

                # iSave = 100
                # if args.search_structure and writer is not None and i % iSave == iSave-1:    # print every iSave mini-batches
                #     if cell_weights is not None:
                #         img = plotsearch.plot(cell_weights)
                #         if img.size > 0:
                #             is_success, buffer = cv2.imencode(".png", img, compression_params)
                #             img_enc = io.BytesIO(buffer).read()
                #             filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                #             s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

                #     imgrad = plotgrads.plot(model)
                #     if imgrad.size > 0:
                #         is_success, buffer = cv2.imencode(".png", imgrad)  
                #         img_enc = io.BytesIO(buffer).read()
                #         filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                #         s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)
                #         # Save calls zero_grads so call it after plotgrads.plot

                #     save(model, s3, s3def, args)

                if args.minimum and i+1 >= test_freq:
                    break

                if profile is not None:
                    profile.step()
            except AssertionError:
            #except:
                print ("Unhandled error in train loop.  Continuing")

            results['batches'] += 1
            if args.minimum and i >= test_freq:
                break

        try:

            writer_path = '{}/{}'.format(args.tensorboard_dir, args.model_dest)

            if cell_weights is not None:
                img = plotsearch.plot(cell_weights)
                if img.size > 0:
                    filename = '{}/{}{:04d}_cw.png'.format(writer_path,args.model_dest, epoch )
                    cv2.imwrite(filename, img)

                    #is_success, buffer = cv2.imencode(".png", img, compression_params)
                    #img_enc = io.BytesIO(buffer).read()
                    # filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                    # s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

            # Plot gradients before saving which clears the gradients
            imgrad = plotgrads.plot(model)
            if imgrad.size > 0:
                filename = '{}/{}{:04d}_gn.png'.format(writer_path,args.model_dest, epoch )
                cv2.imwrite(filename, imgrad)  

                # is_success, buffer = cv2.imencode(".png", imgrad)  
                # img_enc = io.BytesIO(buffer).read()
                # filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                # s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

                convmag = plotconvmag.plot(model)
                if convmag.size > 0:
                    filename = '{}/{}{:04d}_cm.png'.format(writer_path,args.model_dest, epoch )
                    cv2.imwrite(filename, convmag)  

            save(model, s3, s3def, args)
            #lr_scheduler.step()
            #scheduler1.step()
            scheduler2.step()

            filename = '{}/{}.pt'.format(writer_path,args.model_dest)
            save_file(model, filename)

            if args.minimum:
                break

            msg = 'epoch {} step {} model {} training complete'.format(epoch, i, args.model_dest)
            if args.job is True:
                print(msg)
            else:
                tqdm.write(msg)

            if cross_entropy_loss: results['train'][args.model_dest]['cross_entropy_loss']=cross_entropy_loss.item()
            if architecture_loss: results['train'][args.model_dest]['architecture_loss']=architecture_loss.item()
            if prune_loss: results['train'][args.model_dest]['prune_loss']=prune_loss.item()
            if loss: results['train'][args.model_dest]['loss']=loss.item()
            if architecture_reduction: results['train'][args.model_dest]['architecture_reduction']=architecture_reduction.item()
            if training_accuracy: results['train'][args.model_dest]['accuracy'] =  training_accuracy.item()

        except AssertionError:
        #except:
            print ("Unhandled error in epoch reporting.  Continuing")

    #save(model, s3, s3def, args)

    return results

def Test(args, s3, s3def, model, model_vision, loaders, device, results, writer, transforms_vision, profile=None):
    torch.cuda.empty_cache()
    # now = datetime.now()
    # date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    # test_summary = {'date':date_time}

    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset)) 

    if args.test_dir is not None:
        outputdir = '{}/{}'.format(args.test_dir,args.model_class)
        os.makedirs(outputdir, exist_ok=True)
    else:
        outputdir = None

    # model_outputs = {}
    # model_vision_outputs = {}
    
    # model.cells[0].cnn[0].conv.register_forward_hook(activation_hook('model.cells[0].cnn[0].conv', model_outputs))
    # model_vision.conv1.register_forward_hook(activation_hook('model_vision.conv1', model_vision_outputs))

    # model.cells[1].cnn[0].conv.register_forward_hook(activation_hook('model.cells[1].cnn[0].conv', model_outputs))
    # model_vision.layer1[0].conv1.register_forward_hook(activation_hook('model_vision.layer1[0].conv1', model_vision_outputs))

    # model.cells[1].cnn[1].conv.register_forward_hook(activation_hook('model.cells[1].cnn[1].conv', model_outputs))
    # model_vision.layer1[0].conv2.register_forward_hook(activation_hook('model_vision.layer1[0].conv2', model_vision_outputs))

    # model.cells[2].cnn[0].conv.register_forward_hook(activation_hook('model.cells[2].cnn[0].conv', model_outputs))
    # model_vision.layer1[1].conv1.register_forward_hook(activation_hook('model_vision.layer1[1].conv1.', model_vision_outputs))

    # model.cells[2].cnn[1].conv.register_forward_hook(activation_hook('model.cells[2].cnn[1].conv', model_outputs))
    # model_vision.layer1[1].conv2.register_forward_hook(activation_hook('model_vision.layer1[1].conv2', model_vision_outputs))

    # model.cells[2].cnn[2].conv.register_forward_hook(activation_hook('model.cells[2].cnn[2].conv', model_outputs))
    # model_vision.layer1[1].conv3.register_forward_hook(activation_hook('model_vision.layer1[1].conv3', model_vision_outputs))

    # model.cells[2].cnn[2].batchnorm2d.register_forward_hook(activation_hook('model.cells[2].cnn[2].batchnorm2d', model_outputs))
    # model_vision.layer1[1].bn3.register_forward_hook(activation_hook('model_vision.layer1[1].bn3', model_vision_outputs))

    # model.cells[3].cnn[0].conv.register_forward_hook(activation_hook('model.cells[3].cnn[0].conv', model_outputs))
    # model_vision.layer1[2].conv1.register_forward_hook(activation_hook('model_vision.layer1[2].conv1', model_vision_outputs))

    # model.cells[4].cnn[0].conv.register_forward_hook(activation_hook('model.cells[4].cnn[0].conv', model_outputs))
    # model_vision.layer2[0].conv1.register_forward_hook(activation_hook('model_vision.layer2[0].conv1', model_vision_outputs))

    # model.cells[5].cnn[0].conv.register_forward_hook(activation_hook('model.cells[5].cnn[0].conv', model_outputs))
    # model_vision.layer2[1].conv1.register_forward_hook(activation_hook('model_vision.layer2[1].conv1', model_vision_outputs))

    # model.cells[6].cnn[0].conv.register_forward_hook(activation_hook('model.cells[6].cnn[0].conv', model_outputs))
    # model_vision.layer2[2].conv1.register_forward_hook(activation_hook('model_vision.layer2[2].conv1', model_vision_outputs))

    # model.cells[7].cnn[0].conv.register_forward_hook(activation_hook('model.cells[7].cnn[0].conv', model_outputs))
    # model_vision.layer2[3].conv1.register_forward_hook(activation_hook('model_vision.layer2[3].conv1', model_vision_outputs))

    # model.cells[8].cnn[0].conv.register_forward_hook(activation_hook('model.cells[8].cnn[0].conv', model_outputs))
    # model_vision.layer3[0].conv1.register_forward_hook(activation_hook('model_vision.layer3[0].conv1', model_vision_outputs))

    # model.cells[9].cnn[0].conv.register_forward_hook(activation_hook('model.cells[9].cnn[0].conv', model_outputs))
    # model_vision.layer3[1].conv1.register_forward_hook(activation_hook('model_vision.layer3[1].conv1', model_vision_outputs))

    # model.cells[10].cnn[0].conv.register_forward_hook(activation_hook('model.cells[10].cnn[0].conv', model_outputs))
    # model_vision.layer3[2].conv1.register_forward_hook(activation_hook('model_vision.layer3[2].conv1', model_vision_outputs))

    # model.cells[11].cnn[0].conv.register_forward_hook(activation_hook('model.cells[11].cnn[0].conv', model_outputs))
    # model_vision.layer3[3].conv1.register_forward_hook(activation_hook('model_vision.layer3[3].conv1', model_vision_outputs))

    # model.cells[12].cnn[0].conv.register_forward_hook(activation_hook('model.cells[12].cnn[0].conv', model_outputs))
    # model_vision.layer3[4].conv1.register_forward_hook(activation_hook('model_vision.layer3[4].conv1', model_vision_outputs))    

    # model.cells[13].cnn[0].conv.register_forward_hook(activation_hook('model.cells[13].cnn[0].conv', model_outputs))
    # model_vision.layer3[5].conv1.register_forward_hook(activation_hook('model_vision.layer3[5].conv1', model_vision_outputs))

    # model.cells[14].cnn[0].conv.register_forward_hook(activation_hook('model.cells[14].cnn[0].conv', model_outputs))
    # model_vision.layer4[0].conv1.register_forward_hook(activation_hook('model_vision.layer4[0].conv1', model_vision_outputs))

    # model.cells[16].cnn[2].conv.register_forward_hook(activation_hook('model.cells[16].cnn[2].conv', model_outputs))
    # model_vision.layer4[2].conv3.register_forward_hook(activation_hook('model_vision.layer4[2].conv3', model_vision_outputs))

    accuracy = 0.0
    samples = 0
    dtSum = 0.0
    inferTime = []
    top1_correct = []
    # top1_vision_correct = []
    for i, data in tqdm(enumerate(testloader['dataloader']), 
                        total=testloader['batches'], 
                        desc="Test steps", 
                        disable=args.job, 
                        bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
        inputs, labels = data
        #inputs = transforms_vision(inputs)

        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        initial = datetime.now()
        with torch.no_grad():
            outputs = model(inputs)
            classifications = torch.argmax(outputs, 1)

            # outputs_vision = model_vision(inputs)
            # classifications_vision = torch.argmax(outputs_vision, 1)

        dt = (datetime.now()-initial).total_seconds()
        dtSum += dt
        inferTime.append(dt/args.batch_size)

        # mvkeys = list(model_vision_outputs.keys())
        # for i, key in enumerate(model_outputs.keys()):
        #     max_diff = torch.abs(torch.max(model_outputs[key] - model_vision_outputs[mvkeys[i]])).item()
        #     if max_diff > 0:
        #         print('{} != {} diff {}'.format(key, mvkeys[i]))



        top1_step = classifications == labels
        top1_correct.extend(top1_step.cpu().tolist())

        # top1_step_vision = classifications_vision == labels
        # top1_vision_correct.extend(top1_step_vision.cpu().tolist())

        step_accuracy = torch.sum(top1_step)/len(top1_step)
        #step_accuracy_vision = torch.sum(top1_step_vision)/len(top1_step_vision)
        #tqdm.write('test samples={} inferTime={:.5f} step accuracy={:.3f} step_accuracy_vision={:.3f}'.format(len(top1_step), inferTime[-1], step_accuracy, step_accuracy_vision))
        tqdm.write('test samples={} inferTime={:.5f} step accuracy={:.3f}'.format(len(top1_step), inferTime[-1], step_accuracy))
        if writer is not None:
            writer.add_scalar('test/infer', inferTime[-1], results['batches'])
            writer.add_scalar('test/accuracy', step_accuracy, results['batches'])

        if args.minimum and i+1 >= 10:
            break

        if profile is not None:
            profile.step()

    accuracy = np.sum(np.array(top1_correct))/len(top1_correct)
    #accuracy_vision = np.sum(np.array(top1_vision_correct))/len(top1_vision_correct)

    results['test'][args.model_dest] = {
            'accuracy': accuracy.item(),
            #'accuracy_vision': accuracy_vision.item(),
            'minimum time': float(np.min(inferTime)),
            'average time': float(dtSum/len(top1_correct)),
            'num images': len(top1_correct),
        }
    # test_summary['object store'] =s3def
    # test_summary['config'] = args.__dict__
    # if args.ejector is not None and type(args.ejector) != str:
    #     test_summary['config']['ejector'] = args.ejector.value
    # test_summary['system'] = results['system']
    # test_summary['training_results'] = results

    # # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
    # test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
    # training_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)
    # if training_data is None or type(training_data) is not list:
    #     training_data = []
    # training_data.append(test_summary)
    # s3.PutDict(s3def['sets']['test']['bucket'], test_path, training_data)

    # test_url = s3.GetUrl(s3def['sets']['test']['bucket'], test_path)
    # print("Test results {}".format(test_url))
    # tqdm.write('test results={}'.format(yaml.dump(test_summary['results'], default_flow_style=False) ) )

    # if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.tb_dest is not None and len(args.tb_dest) > 0):
    #     writer_path = '{}/{}/testresults.yaml'.format(args.tensorboard_dir, args.model_dest)
    #     WriteDict(test_summary, writer_path)

    # results['test'][args.model_dest] = test_summary['results']
    return results


def Prune(args, s3, s3def, model, loaders, results):
    torch.cuda.empty_cache()
    model.ApplyStructure()

    parameters_after_prune, flops_after_prune = ModelSize(args, model, loaders)

    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.tb_dest is not None and len(args.tb_dest) > 0):
        writer_path = '{}/{}'.format(args.tensorboard_dir, args.model_dest)
        filename = '{}/{}.pt'.format(writer_path,args.model_dest)
        save_file(model, filename)

    save(model, s3, s3def, args)
    results['prune'][args.model_dest] = {
                        'final parameters':parameters_after_prune, 
                        'initial parameters' : results['initial_parameters'], 
                        'final/intial params': parameters_after_prune/results['initial_parameters'], 
                        'final FLOPS': flops_after_prune, 
                        'initial FLOPS': results['initial_flops'], 
                        'final/intial FLOPS':flops_after_prune/results['initial_flops'] }
    print('{} prune results {}'.format(args.model_dest, yaml.dump(results['prune'], default_flow_style=False)))

    return results

def onnx(model, s3, s3def, args, input_channels):
    torch.cuda.empty_cache()
    import torch.onnx as torch_onnx

    dummy_input = torch.randn(args.batch_size, input_channels, args.height, args.width, device='cuda')
    input_names = ["image"]
    output_names = ["segmentation"]
    oudput_dir = args.tensorboard_dir
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


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def main(args):

    if args.output_dir:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    utils.init_distributed_mode(args)
    #print(args)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    config = ReadDict(args.config)
    version_str = VersionString(config)
    #print('{} version {}'.format(__file__, version_str))

    versions = {
        'platform':str(platform.platform()),
        'python':str(platform.python_version()),
        'numpy': str(np.__version__),
        'torch': str(torch.__version__),
        'OpenCV': str(cv2.__version__),
        'pymlutil': str(pymlutil_version.__version__),
        'cell2d':version_str
    }

    results = {
            'batches': 0,
            'initial_parameters': None,
            'initial_flops': None,
            'runs': {},
            'load': {},
            'prune': {},
            'store': {},
            'train': {},
            'test': {},
        }

    results['runs'][args.model_dest] = {
            'arguments': args.__dict__,
            'versions': versions,
        }

    results['runs'][args.model_dest]['arguments']['ejector'] = args.ejector.value # Convert from enum to string
    print('{}'.format(yaml.dump(results, default_flow_style=False) ))

    #torch.autograd.set_detect_anomaly(True)
    s3, _, s3def = Connect(args.credentails, s3_name=args.s3_name)

    results['runs'][args.model_dest]['store'] = s3def

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
                                       offset=args.augment_translate_x,
                                       augment_noise=args.augment_noise,
                                       width=args.width, height=args.height)

    elif args.dataset == 'imagenet':
        loaders = CreateImagenetLoaders(s3, s3def, 
                                        args.obj_imagenet, 
                                        args.dataset_path+'/imagenet', 
                                        crop_width=args.width, 
                                        crop_height=args.height, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        cuda = args.cuda,
                                        rotate=args.augment_rotation, 
                                        scale_min=args.augment_scale_min, 
                                        scale_max=args.augment_scale_max, 
                                        offset=args.augment_translate_x,
                                        augment_noise=args.augment_noise,
                                        augment=False,
                                        normalize=True
                                       )
    else:
        raise ValueError("Unupported dataset {}".format(args.dataset))

    # Load number of previous batches to continue tensorboard from previous training
    prevresultspath = None
    print('prevresultspath={}'.format(args.prevresultspath))
    if args.prevresultspath and len(args.prevresultspath) > 0:
        prevresults = ReadDict(args.prevresultspath)

        if prevresults is not None:
            results.update(prevresults)
            if 'batches' in prevresults:
                print('found prevresultspath={}'.format(yaml.dump(prevresults, default_flow_style=False)))
                results['batches'] = prevresults['batches']
            if 'initial_parameters' in prevresults:
                results['initial_parameters'] = prevresults['initial_parameters']
                results['initial_flops'] = prevresults['initial_flops']

    classify, results, model_vision, transforms_vision = load(s3, s3def, args, loaders, results)

    # Prune with loaded parameters than apply current search_structure setting
    classify.ApplyParameters(weight_gain=args.weight_gain, 
                            sigmoid_scale=args.sigmoid_scale,
                            feature_threshold=args.feature_threshold,
                            search_structure=args.search_structure, 
                            convMaskThreshold=args.convMaskThreshold, 
                            k_prune_sigma=args.k_prune_sigma,
                            search_flops=args.search_flops,
                            batch_norm=args.batch_norm)


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

        # if args.write_vision_graph:
        #     WriteModelGraph(args, writer, model_vision, loaders)
        # else:
        WriteModelGraph(args, writer, classify, loaders)

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
                results = Test(args, s3, s3def, classify, model_vision, loaders, device, results, writer, transforms_vision, prof)
        else:
            results = Test(args, s3, s3def, classify, model_vision, loaders, device, results, writer, transforms_vision)

    if args.onnx:
        onnx(classify, s3, s3def, args, loaders[0]['in_channels'])

    # if args.resultspath is not None and len(args.resultspath) > 0:
    #     WriteDict(results, args.resultspath)

    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.tb_dest is not None and len(args.tb_dest) > 0):
        tb_results = os.path.join(args.tensorboard_dir, args.resultspath)
        WriteDict(results, tb_results)

        tb_path = os.path.join(s3def['sets']['model']['prefix'],args.model_class,args.tb_dest)
        print('Write tensorboard to s3 {}/{}'.format(s3def['sets']['test']['bucket'], args.tensorboard_dir))
        s3.PutDir(s3def['sets']['test']['bucket'], args.tensorboard_dir, tb_path )

    # LogTest(args, s3, s3def, results)

    print('Finished {}'.format(args.model_dest ))
    print(yaml.dump(results, default_flow_style=False))
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

        Connect to vscode "Python: Remote" configuration
        '''

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = main(args)
    sys.exit(result)

