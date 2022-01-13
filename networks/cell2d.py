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
import torch
from torch._C import parse_ir
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import cv2

sys.path.insert(0, os.path.abspath(''))
from utils.torch_util import count_parameters, model_stats, model_weights
from utils.jsonutil import ReadDictJson
from utils.s3 import s3store, Connect
from torch.utils.tensorboard import SummaryWriter
from networks.totalloss import TotalLoss

# Inner neural architecture cell repetition structure
# Process: Con2d, optional batch norm, optional ReLu

def GaussianBasis(i, depth, r=1.0):
    return torch.exp(-1*(r*(i-depth))*(r*(i-depth))) # torch.square not supported by torch.onnx

def NormGausBasis(len, i, depth, r=1.0):
        den = 0.0
        num = 0.0
        for j in range(len):
            bias = GaussianBasis(j,depth,r)
            if j==i:
                num=bias
            den = den + bias
        return num/den

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
                 definition=None,
                 dropout_rate=0.2,
                 sigmoid_scale = 5.0, # Channel sigmoid scale fatctor
                 search_structure=True,
                 ):
        super(ConvBR, self).__init__()
        self.in_channels = in_channels
        if out_channels < 1:
            raise ValueError("out_channels must be > 0")
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.relu = relu
        self.weight_gain = weight_gain
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

        if definition is not None:
            for key in definition:
                self[key] = definition[key]

        
        self.channel_scale = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float))

        if type(kernel_size) == int:
            padding = kernel_size // 2 # dynamic add padding based on the kernel_size
        else:
            padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if self.batch_norm:
            self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.sigmoid = nn.Sigmoid()
        self.total_trainable_weights = model_weights(self)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self._initialize_weights()
        norm = torch.linalg.norm(self.conv.weight, dim=(1,2,3))/np.sqrt(np.product(self.conv.weight.shape[1:]))
        print('ConvBR initialized weights {}'.format(norm))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.batch_norm and self.batchnorm2d and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def definition(self):
        definition_dict = {
            'in_channelss': self.in_channels,
            'out_channels': self.out_channels, 
            'batch_norm': self.batch_norm,
            'relu': self.relu, 
            'weight_gain': self.weight_gain,
            'convMaskThreshold': self.convMaskThreshold,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
        }

        #dfn = deepcopy(self.__dict__)
        #dfn['depth'] = self.depth.item()

        return definition_dict

    def forward(self, x, isTraining=False):
        if isTraining:
            x = self.dropout(x)
        x = self.conv(x)
        if self.search_structure: #scale channels based on self.channel_scale
            weight_scale = self.sigmoid(self.sigmoid_scale*self.channel_scale)[None,:,None,None]
            x *= weight_scale

        if self.batch_norm:
            x = self.batchnorm2d(x)
        if self.relu:
            x = F.relu(x, inplace=True)

        return x

    def ArchitectureWeights(self):
        weight_scale = self.sigmoid(self.sigmoid_scale*self.channel_scale)
        norm = torch.linalg.norm(self.conv.weight, dim=(1,2,3))/np.sqrt(np.product(self.conv.weight.shape[1:]))
        conv_weights = (torch.tanh(self.weight_gain*norm)+weight_scale)/2.0

        # Keep sum as [1] tensor so subsiquent concatination works
        architecture_weights = (self.total_trainable_weights/ self.out_channels) * conv_weights.sum_to_size((1))

        return architecture_weights, self.total_trainable_weights, conv_weights

    # Remove specific network dimensions
    # remove dimension where inDimensions and outDimensions arrays are 0 for channels to be removed
    def ApplyStructure(self, in_channel_mask=None, out_channel_mask=None, weight_mask=True, msg=None):
        if in_channel_mask is not None:
            if len(in_channel_mask) == self.in_channels:
                self.conv.weight.data = self.conv.weight[:, in_channel_mask!=0]
                self.in_channels = len(in_channel_mask[in_channel_mask!=0])
            else:
                raise ValueError("len(in_channel_mask)={} must be equal to self.in_channels={}".format(len(in_channel_mask), self.in_channels))

        # Convolution norm gain mask
        #print("ApplyStructure convolution norm {}".format(torch.linalg.norm(self.conv.weight, dim=(1,2,3))))
        if weight_mask:
            norm = torch.linalg.norm(self.conv.weight, dim=(1,2,3))/np.sqrt(np.product(self.conv.weight.shape[1:]))
            conv_mask = torch.tanh(self.weight_gain*norm)
            conv_mask += self.sigmoid(self.sigmoid_scale*self.channel_scale)
            conv_mask = conv_mask/2.0 > self.convMaskThreshold
        else:
            conv_mask = torch.ones(self.conv.weight.shape[0], dtype=torch.bool, device=self.conv.weight.device)


        if out_channel_mask is not None:
            if len(out_channel_mask) == self.out_channels:
                # If either mask is false, then the convolution is removed (not nand)
                conv_mask = torch.logical_not(torch.logical_or(torch.logical_not(conv_mask), torch.logical_not(out_channel_mask)))
                
            else:
                raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))

        pruned_convolutions = len(conv_mask[conv_mask==False])
        prev_colutions = len(conv_mask)

        self.conv.bias.data = self.conv.bias[conv_mask!=0]
        self.conv.weight.data = self.conv.weight[conv_mask!=0]

        if self.batch_norm:
            self.batchnorm2d.bias.data = self.batchnorm2d.bias.data[conv_mask!=0]
            self.batchnorm2d.weight.data = self.batchnorm2d.weight.data[conv_mask!=0]
            self.batchnorm2d.running_mean = self.batchnorm2d.running_mean[conv_mask!=0]
            self.batchnorm2d.running_var = self.batchnorm2d.running_var[conv_mask!=0]

        self.out_channels = len(conv_mask[conv_mask!=0])

        if msg is None:
            prefix = "ConvBR::ApplyStructure"
        else:
            prefix = "ConvBR::ApplyStructure {}".format(msg)
        print("{} {}={}/{} in_channels={} out_channels={}".format(prefix, pruned_convolutions/prev_colutions, pruned_convolutions, prev_colutions, self.in_channels, self.out_channels))

        return conv_mask


DefaultMaxDepth = 1
class Cell(nn.Module):
    def __init__(self,
                 in1_channels, 
                 in2_channels = 0,
                 batch_norm=False, 
                 relu=True,
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1,
                 bias=True, 
                 padding_mode='zeros',
                 residual=True,
                 is_cuda=False,
                 feature_threshold=0.01,
                 search_structure=True,
                 cell_convolution=DefaultMaxDepth,
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 convolutions=[{'out_channels':64, 'kernel_size': 3, 'stride': 1, 'dilation': 1}],
                 definition=None,
                 dropout_rate = 0.2,
                 sigmoid_scale = 5.0
                 ):
                
        super(Cell, self).__init__()

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.batch_norm = batch_norm
        self.relu = relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.residual = residual
        self.is_cuda = is_cuda
        self.feature_threshold = feature_threshold
        self.search_structure = search_structure
        self.cell_convolution = nn.Parameter(torch.tensor(cell_convolution, dtype=torch.float))
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold
        self.convolutions = convolutions
        self.dropout_rate = dropout_rate
        self.sigmoid_scale=sigmoid_scale

        if definition is not None:
            for key in definition:
                self.__dict__[key] = definition[key]
                
            if 'cell_convolution' in definition:
                self.cell_convolution = nn.Parameter(torch.tensor(definition['cell_convolution'], dtype=torch.float))

        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convoutions uses out_channels as chanels

        convdfn = None
        if definition is not None and 'conv_size' in definition:
            convdfn = definition['conv_size']

        src_channels = self.in1_channels+self.in2_channels

        totalStride = 1
        totalDilation = 1

        for i, convdev in enumerate(convolutions):
            convdfn = None
            if definition is not None and 'cnn' in definition and i < len(definition['cnn']):
                convdfn = definition['cnn'][i]

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
                definition=convdfn,
                dropout_rate=self.dropout_rate,
                search_structure=self.search_structure,
                sigmoid_scale = self.sigmoid_scale)
            self.cnn.append(conv)

            src_channels = convdev['out_channels']
            totalStride *= convdev['stride']
            totalDilation *= convdev['dilation']
 

        self.conv_residual = ConvBR(self.in1_channels+self.in2_channels, self.convolutions[-1]['out_channels'], 
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
            definition=convdfn,
            dropout_rate=self.dropout_rate,
            search_structure=False)

        self._initialize_weights()
        self.total_trainable_weights = model_weights(self)


    def definition(self):
        definition_dict = {
            'in1_channels': self.in1_channels, 
            'in2_channels': self.in2_channels,
            'batch_norm': self.batch_norm,
            'relu': self.relu,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'padding_mode': self.padding_mode,
            'residual': self.residual,
            'is_cuda': self.is_cuda,
            'feature_threshold': self.feature_threshold,
            'search_structure': self.search_structure,
            'cell_convolution': self.cell_convolution.item(),
            'total_trainable_weights': self.total_trainable_weights,
        }

        # definition_dict['conv_size'] = self.conv_size.definition()
        definition_dict['cnn'] = []
        if self.cnn is not None:
            for conv in self.cnn:
                definition_dict['cnn'].append(conv.definition())
        #definition_dict['conv1x1'] = self.conv1x1.definition()

        #dfn = deepcopy(self.__dict__)
        #dfn['cell_convolution'] = self.cell_convolution.item()

        return definition_dict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def ApplyStructure(self, in1_channel_mask=None, in2_channel_mask=None, msg=None):

        # Reduce channels
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
            in_channel_mask = torch.ones((self.in1_channels+self.in2_channels), dtype=torch.int32)
        
        if self.cnn is not None:
            out_channel_mask = in_channel_mask

            for i, cnn in enumerate(self.cnn):
                layermsg = "convolution {}/{}".format(i, len(self.cnn))

                if msg is not None:
                    layermsg = "{} {}".format(msg, layermsg)
                    
                out_channel_mask = cnn.ApplyStructure(in_channel_mask=out_channel_mask, msg=layermsg)
        else:
            out_channel_mask = None

        layermsg = "cell residual"
        if msg is not None:
            layermsg = "{} {}".format(msg, layermsg)

        self.conv_residual.ApplyStructure(in_channel_mask=in_channel_mask, out_channel_mask=out_channel_mask, weight_mask=False, msg=layermsg)

        if self.convMaskThreshold > torch.tanh(torch.abs(self.weight_gain*self.cell_convolution)):
            self.cnn = None
            #if self.convolutions[-1]['out_channels'] == self.in1_channels+self.in2_channels:
            #    self.conv_residual = None
            
            print('Pruning cell weight={} cell weights={} in1_channels={} in2_channels={} out_channels={}'.format(self.cell_convolution, self.total_trainable_weights, self.in1_channels, self.in2_channels, self.convolutions[-1]['out_channels']))
        

        return out_channel_mask


    def forward(self, in1, in2 = None, isTraining=False):
        if in2 is not None:
            u = torch.cat((in1, in2), dim=1)
        else:
            u = in1

        # Resizing convolution
        if self.conv_residual:
            residual = self.conv_residual(u, isTraining)
            #residual = self.conv_residual(u)

        if self.cnn is not None:
            x = u
            for i, l in enumerate(self.cnn):
                x = self.cnn[i](x, isTraining) 

            if self.conv_residual:
                y = residual + x*torch.tanh(torch.abs(self.weight_gain*self.cell_convolution))
        elif self.conv_residual:
            y = residual
        else:
            y = u

        return y

    def ArchitectureWeights(self):
        architecture_weights = []

        layer_weights = []
        cell_weight = []
        prune_weight = torch.tanh(torch.abs(self.weight_gain*self.cell_convolution))
        prune_weight = torch.ones_like(prune_weight)

        # Not pruning residual weights 
        #if self.conv_residual is not None:
        #    layer_weight, _, conv_weights  = self.conv_residual.ArchitectureWeights()
        #    cell_weight.append(conv_weights)
        #    architecture_weights += layer_weight 

        if self.cnn is not None:
            for i, l in enumerate(self.cnn): 
                layer_weight, _, conv_weights  = l.ArchitectureWeights()
                architecture_weights.append(layer_weight)
                cell_weight.append(conv_weights)

            architecture_weights = torch.cat(architecture_weights)
            architecture_weights = architecture_weights.sum_to_size((1))
        else:
            architecture_weights = torch.zeros((1), device=self.cell_convolution.device)

        cell_weights = {'prune_weight':prune_weight, 'cell_weight':cell_weight}

        # Enable "architecture_weights *= prune_weight" when convolution weights are successfully optimized
        #architecture_weights *= prune_weight

        return architecture_weights, self.total_trainable_weights, cell_weights

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
            in_channel_mask = torch.ones(self.in_channels, dtype=torch.int32)

        if out_channel_mask is not None:
            if len(out_channel_mask) != self.out_channels:
                  raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))
        else: # Do not reduce input channels.  
            out_channel_mask = torch.ones(self.out_channels, dtype=torch.int32)

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

def ResnetCells(size = Resnet.layers_50):
    resnetCells = []
    network_channels = [64, 128, 256, 512]
    
    sizes = {
        'layers_18': [2, 2, 2, 2], 
        'layers_34': [3, 4, 6, 3], 
        'layers_50': [3, 4, 6, 3], 
        'layers_101': [3, 4, 23, 3], 
        'layers_152': [3, 8, 36, 3]
        }
    bottlenecks = {
        'layers_18': False, 
        'layers_34': False, 
        'layers_50': True, 
        'layers_101': True, 
        'layers_152': True
        }

    resnet_cells = []
    block_sizes = sizes[size.name]
    bottleneck = bottlenecks[size.name]
    for i, layer_size in enumerate(block_sizes):
        block_channels = network_channels[i]
        for j in range(layer_size):
            stride = 1
            # Downsample by setting stride to 2 on the first layer of each block
            if i != 0 and j == 0:
                stride = 2
            cell = []
            if bottleneck:
                cell.append({'out_channels':network_channels[i], 'kernel_size': 1, 'stride': stride, 'dilation': 1})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': 1, 'dilation': 1})
                cell.append({'out_channels':4*network_channels[i], 'kernel_size': 1, 'stride': 1, 'dilation': 1})
            else:
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': stride, 'dilation': 1})
                cell.append({'out_channels':network_channels[i], 'kernel_size': 3, 'stride': 1, 'dilation': 1})
            resnetCells.append(cell)
        
    return resnetCells


class Classify(nn.Module):
    def __init__(self, convolutions, 
    is_cuda=False, source_channels = 3, out_channels = 10, initial_channels=16, 
    batch_norm=True, weight_gain=11, convMaskThreshold=0.5, definition=None, 
    dropout_rate=0.2, search_structure = True, sigmoid_scale=5.0,):
        super().__init__()
        self.is_cuda = is_cuda
        self.source_channels = source_channels
        self.out_channels = out_channels
        self.initial_channels = initial_channels
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.search_structure = search_structure
        self.sigmoid_scale = sigmoid_scale

        if definition is not None:
            for key in definition:
                self.__dict__[key] = definition[key]
                
        self.cells = torch.nn.ModuleList()
        in_channels = self.source_channels

        for i, cell_convolutions in enumerate(convolutions):

            convdfn = None
            if definition is not None and 'cells' in definition:
                if len(definition['cells']) > i:
                    convdfn = definition['cells'][i] 

            cell = Cell(in1_channels=in_channels, 
                batch_norm=self.batch_norm,
                is_cuda=self.is_cuda,  
                weight_gain = self.weight_gain,
                convMaskThreshold = self.convMaskThreshold,
                convolutions=cell_convolutions,  
                definition=convdfn,
                dropout_rate=self.dropout_rate, 
                search_structure=self.search_structure, 
                sigmoid_scale=self.sigmoid_scale)
            in_channels = cell_convolutions[-1]['out_channels']
            self.cells.append(cell)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FC(in_channels, self.out_channels)

        self.total_trainable_weights = model_weights(self)

        self.fc_weights = model_weights(self.fc)

    def definition(self):
        definition_dict = {
            'source_channels': self.source_channels,
            'out_channels': self.out_channels,
            'initial_channels': self.initial_channels,
            'is_cuda': self.is_cuda,
            #'min_search_depth': self.min_search_depth,
            #'max_search_depth': self.max_search_depth,
            #'max_cell_steps': self.max_cell_steps,
            #'channel_multiple': self.channel_multiple,
            #'batch_norm': self.batch_norm,
            #'search_structure': self.search_structure,
            #'cell_convolution': self.cell_convolution.item(),
            'cells': [],
        }

        for ed in self.cells:
            definition_dict['cells'].append(ed.definition())

        architecture_weights, total_trainable_weights, cell_weights = self.ArchitectureWeights()
        definition_dict['architecture_weights']= architecture_weights.item()
        definition_dict['total_trainable_weights']= total_trainable_weights

        return definition_dict

    def ApplyStructure(self):
        in_channel_mask = None
        for i, cell in enumerate(self.cells):
            layer_msg = 'Cell {}'.format(i)
            out_channel_mask = cell.ApplyStructure(in1_channel_mask=in_channel_mask, msg=layer_msg)
            in_channel_mask = out_channel_mask

        self.fc.ApplyStructure(in_channel_mask=in_channel_mask)

    def forward(self, x, isTraining=False):
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-learning_rate', type=float, default=0.01, help='Training learning rate')
    parser.add_argument('-batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='Classification')
    parser.add_argument('-model_class', type=str,  default='CIFAR10')
    parser.add_argument('-model_src', type=str,  default=None)
    parser.add_argument('-model_dest', type=str, default='nas_20220113_00')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-k_structure', type=float, default=1.0e-3, help='Structure minimization weighting fator')
    parser.add_argument('-target_structure', type=float, default=1.0e-1, help='Structure minimization weighting fator')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout_rate', type=float, default=0.3, help='Dropout probabability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convoluiton channels weights')

    parser.add_argument('-prune', type=bool, default=False)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-infer', type=bool, default=True)
    parser.add_argument('-search_structure', type=bool, default=True)
    parser.add_argument('-onnx', type=bool, default=True)

    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='/store/test/nassegtb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir /store/test/nassegtb --bind_all')

    parser.add_argument('-description', type=json.loads, default='{"description":"Cell 2D NAS classification"}', help='Run description')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152], default=50, help='Run description')

    args = parser.parse_args()
    return args

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model, out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)
    #s3.PutDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), model.definition())

class PlotSearch():
    def __init__(self, network, save_image = 'class_weights.png', save_video = 'class_weights.mp4', title = 'Architecture Weights', colormapname = 'jet', lenght = 5, width=7.5, height=20, dpi=1200, fps=2 ):

        self.save_image = save_image
        self.save_video = save_video
        self.title = title
        self.colormapname = colormapname
        self.lenght = int(lenght)
        self.dpi = dpi
        self.cm = plt.get_cmap(colormapname)
        self.clear_frames = True
        self.fps = fps
        self.thickness=1

        self.fig, self.ax = plt.subplots(figsize=(width,height), dpi=self.dpi) # note we must use plt.subplots, not plt.subplot 

        architecture_weights, total_trainable_weights, cell_weights = network.ArchitectureWeights()
        self.height = 0
        self.width = 0
        for i,  cell, in enumerate(cell_weights):
            for j, step in enumerate(cell['cell_weight']):
                self.width += self.lenght
                self.height = max(self.height, self.thickness*len(step))

        # Output video writer
        metadata = dict(title='self.title', artist='Matplotlib', comment='Plot neural architecture search')
        self.writer = FFMpegWriter(fps=self.fps, metadata=metadata)
        self.writer.setup(self.fig, self.save_video)
    def __del__(self):
        self.finish()

    def plot_weights(self, weights, index = None):

        img = np.zeros([self.height,self.width,3]).astype(np.uint8)
        
        #self.ax.clear()
        
        if index:
            title = '{} {}'.format(self.title, index)
        else:
            title = self.title

        #self.ax.set_title(title)
        #xMax = 1.0
        #yMax = 1.0
         

        for i,  cell, in enumerate(weights):
            prune_weight = cell['prune_weight'].cpu().detach().numpy()
            for j, step in enumerate(cell['cell_weight']):
                x = i*self.lenght*len(cell['cell_weight'])+j*self.lenght

                for k, gain in enumerate(step.cpu().detach().numpy()):
                    
                    y = int(k*self.thickness+self.thickness/2)
                    start_point = (x,y)
                    end_point=(x+self.lenght,y)

                    conv_gain = prune_weight*gain
                    color = 255*np.array(self.cm(conv_gain))
                    color = color.astype('uint8')
                    colorbgr = (int(color[2]), int(color[1]), int(color[0]))

                    cv2.line(img,start_point,end_point,colorbgr,self.thickness)
                    #cv2.line(img, start_point=start_point, end_point=end_point, color=colorbgr, thickness=thickness)
                    
                    #circleXMax = i*self.lenght*len(cell['cell_weight'])+(j+1)*self.lenght
                    #circleYMax = (k+1)*self.lenght

                    #if circleXMax > xMax:
                    #    xMax = circleXMaxTotal Trainable Params: 27620940

        #self.fig.savefig(self.save_image)
        cv2.imwrite(self.save_image, img)
        #self.writer.grab_frame()

    def finish(self):
        print('PlotSearch finish')
        #self.writer.finish()


# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    import torchvision
    import torchvision.transforms as transforms

    system = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'numpy version': sys.modules['numpy'].__version__,
    }

    print('Cell Test system={}'.format(system))

    torch.autograd.set_detect_anomaly(True)

    s3, creds, s3def = Connect(args.credentails)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)



    device = "cpu"
    pin_memory = False
    if args.cuda:
        device = "cuda"
        pin_memory = True

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainingset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
    train_batches=int(trainingset.__len__()/args.batch_size)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
    test_batches=int(testset.__len__()/args.batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_freq = int(math.ceil(train_batches/test_batches))

    # Create classifier
    if(args.model_src and args.model_src != ''):
        #modeldict = s3.GetDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            classify = torch.load(io.BytesIO(modelObj))
        else:
            print('Failed to load model {}. Exiting.'.format(args.model_src))
            return -1
    else:
        # Create Default classifier
        resnetCells = ResnetCells(Resnet(args.resnet_len))
        classify = Classify(convolutions=resnetCells, 
                            is_cuda=args.cuda, 
                            weight_gain=args.weight_gain, 
                            dropout_rate=args.dropout_rate, 
                            search_structure=args.search_structure, 
                            sigmoid_scale=args.sigmoid_scale,
                            batch_norm = args.batch_norm)

    #sepcificy device for model
    classify.to(device)
    
    total_parameters = count_parameters(classify)

    if args.prune:
        classify.ApplyStructure()
        reduced_parameters = count_parameters(classify)
        print('Reduced parameters {}/{} = {}'.format(reduced_parameters, total_parameters, reduced_parameters/total_parameters))
        save(classify, s3, s3def, args)

    # Define a Loss function and optimizer
    target_structure = torch.as_tensor([args.target_structure], dtype=torch.float32)
    criterion = TotalLoss(args.cuda, k_structure=args.k_structure, target_structure=target_structure, search_structure=args.search_structure)
   #optimizer = optim.SGD(classify.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(classify.parameters(), lr=args.learning_rate)
    plotsearch = PlotSearch(classify)
    iSample = 0

        # Train
    if args.train:
        for epoch in tqdm(range(args.epochs), desc="Train epochs"):  # loop over the dataset multiple times
            iTest = iter(testloader)

            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader), total=trainingset.__len__()/args.batch_size, desc="Train steps"):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = classify(inputs, isTraining=True)
                loss, cross_entropy_loss, architecture_loss, arcitecture_reduction, cell_weights  = criterion(outputs, labels, classify)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                writer.add_scalar('loss/train', loss, iSample)
                writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, iSample)
                writer.add_scalar('architecture_loss/train', architecture_loss, iSample)
                writer.add_scalar('arcitecture_reduction/train', arcitecture_reduction, iSample)

                if i % test_freq == test_freq-1:    # Save image and run test

                    classifications = torch.argmax(outputs, 1)
                    results = (classifications == labels).float()
                    training_accuracy = torch.sum(results)/len(results)

                    data = next(iTest)
                    inputs, labels = data

                    if args.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    optimizer.zero_grad()
                    outputs = classify(inputs)
                    classifications = torch.argmax(outputs, 1)
                    results = (classifications == labels).float()
                    test_accuracy = torch.sum(results)/len(results)

                    loss, cross_entropy_loss, architecture_loss, arcitecture_reduction, cell_weights  = criterion(outputs, labels, classify)

                    writer.add_scalar('accuracy/train', training_accuracy, iSample)
                    writer.add_scalar('accuracy/test', test_accuracy, iSample)
                    writer.add_scalar('loss/test', loss, iSample)
                    writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, iSample)
                    writer.add_scalar('architecture_loss/test', architecture_loss, iSample)
                    writer.add_scalar('arcitecture_reduction/train', arcitecture_reduction, iSample)

                    running_loss /=test_freq
                    tqdm.write('Test [{}, {:06f}] training accuracy={:03f} test accuracy={:03f} training loss={:0.5e}, test loss={:0.5e} arcitecture_reduction: {:0.5e}'.format(
                        epoch + 1, i + 1, training_accuracy, test_accuracy, running_loss, loss, arcitecture_reduction))
                    running_loss = 0.0

                    plotsearch.plot_weights(cell_weights)

                iSave = 2000
                if i % iSave == iSave-1:    # print every 20 mini-batches
                    save(classify, s3, s3def, args)

                if args.fast and i+1 >= test_freq:
                    break

                iSample += 1

            save(classify, s3, s3def, args)

    print('Finished cell2d Test')
    return 0

if __name__ == '__main__':
    import argparse
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
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    result = Test(args)
    sys.exit(result)

