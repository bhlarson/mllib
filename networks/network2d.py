#!/usr/bin/python3
from dis import dis
import math
import os
import sys
import copy
import io
import json
import platform
import time
import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard import program
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
import cv2
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.abspath(''))
from networks.cell2d import Cell, GaussianBasis, NormGausBasis, PlotSearch, PlotGradients
from utils.torch_util import count_parameters, model_stats, model_weights
from utils.jsonutil import ReadDict, WriteDict
from utils.s3 import s3store, Connect
from datasets.cocostore import CocoDataset
from datasets.imstore import ImagesDataset
from utils.metrics import similarity, jaccard, DatasetResults
from networks.totalloss import TotalLoss, FenceSitterEjectors
from utils.functions import GaussianBasis, SigmoidScale, Exponential


class Network2d(nn.Module):
    def __init__(self, 
                 out_channels=1, 
                 source_channels=3, 
                 initial_channels=64, 
                 device=torch.device("cpu"), 
                 unet_depth=5, 
                 max_cell_steps=6, 
                 channel_multiple=2, 
                 batch_norm=False, 
                 cell=Cell, 
                 search_structure=True,
                 residual = False,
                 dropout = False,
                 feature_threshold=0.5,
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 dropout_rate = 0.2,
                 sigmoid_scale = 5.0,
                 k_prune_sigma = 0.33):
        super(Network2d, self).__init__()

        self.unet_depth = unet_depth
        self.out_channels = out_channels
        self.source_channels = source_channels
        self.initial_channels = initial_channels
        self.device = device
        self.cell = cell
        self.max_cell_steps = max_cell_steps
        self.channel_multiple = channel_multiple
        self.batch_norm = batch_norm

        self.cells = torch.nn.ModuleList()
        self.upsample = torch.nn.ModuleList()
        self.final_conv = torch.nn.ModuleList()
        self.search_structure = search_structure
        self.residual = residual
        self.dropout = dropout

        self.feature_threshold=feature_threshold
        self.weight_gain = weight_gain
        self.convMaskThreshold=convMaskThreshold
        self.dropout_rate = dropout_rate
        self.sigmoid_scale = sigmoid_scale
        self.k_prune_sigma = k_prune_sigma

        encoder_channels = self.initial_channels
        prev_encoder_chanels = self.source_channels
        feedforward_chanels = []

        convolutions=[{'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True},
                        {'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True}]
        for i in range(self.unet_depth-1):
            for convolution in convolutions:
                convolution['out_channels'] = encoder_channels

            cell = self.cell(prev_encoder_chanels, 
                             batch_norm=self.batch_norm, 
                             device=self.device,
                             convolutions=convolutions,
                             search_structure=self.search_structure,
                             residual=self.residual,
                             dropout=self.dropout,
                             dropout_rate=self.dropout_rate, 
                             sigmoid_scale=self.sigmoid_scale, 
                             feature_threshold=self.feature_threshold, 
                             k_prune_sigma=self.k_prune_sigma)
            self.cells.append(cell)

            feedforward_chanels.append(encoder_channels)
            prev_encoder_chanels = encoder_channels
            encoder_channels = int(self.channel_multiple*encoder_channels)

        out_encoder_channels = int(encoder_channels/self.channel_multiple)

        for i in range(self.unet_depth):
            if i == 0:
                feedforward = 0
            else:
                feedforward = feedforward_chanels[-i]

            if i == self.unet_depth-1:
                out_channels = self.out_channels
                final_kernel_size = 1
                final_stride = 1
                conv_transpose = False

            else:
                out_channels =out_encoder_channels
                final_kernel_size = 2
                final_stride = 2
                conv_transpose = True

            convolutions=[{'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True},
                          {'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True},
                          {'out_channels':out_channels, 'kernel_size': final_kernel_size, 'stride': final_stride, 'dilation': 1, 'search_structure':True, 'conv_transpose':conv_transpose}]

            cell = self.cell(prev_encoder_chanels, 
                             feedforward,
                             batch_norm=self.batch_norm,
                             device=self.device,
                             convolutions=convolutions,
                             search_structure=self.search_structure,
                             residual=self.residual,
                             dropout=self.dropout,
                             dropout_rate=self.dropout_rate, 
                             sigmoid_scale=self.sigmoid_scale, 
                             feature_threshold=self.feature_threshold,
                             k_prune_sigma=self.k_prune_sigma)
            self.cells.append(cell)

            prev_encoder_chanels = out_encoder_channels
            encoder_channels = int(encoder_channels/self.channel_multiple)
            out_encoder_channels = int(encoder_channels/self.channel_multiple)

        self.pool = nn.MaxPool2d(2, 2)

    def ApplyParameters(self, search_structure=None, convMaskThreshold=None, dropout=None, 
                        weight_gain=None, sigmoid_scale=None, feature_threshold=None,
                        k_prune_sigma=None): # Apply a parameter change
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
        for cell in self.cells:
            cell.ApplyParameters(search_structure=search_structure, dropout=dropout, convMaskThreshold=convMaskThreshold,
                                 weight_gain=weight_gain, sigmoid_scale=sigmoid_scale, feature_threshold=feature_threshold,
                                 k_prune_sigma=k_prune_sigma)

    def forward(self, x):
        feed_forward = []
        enc_len = math.floor(len(self.cells)/2.0)
        iDecode = enc_len

        # Encoder
        for i in range(enc_len):
            x = self.cells[i](x)
            feed_forward.append(x)
            if x is not None:
                x = self.pool(x)

        if (len(self.cells) % 2) != 0:
            x = self.cells[enc_len](x)
            iDecode += 1
        else:
            x = None

        # Decoder
        for i in range(enc_len):
            x = self.cells[iDecode+i](x, feed_forward[-(i+1)])

        return x

    def ApplyStructureConvTranspose2d(self, conv, in_channels=None, out_channels=None):

        if in_channels is not None:
            if len(in_channels) == conv.in_channels:
                conv.weight.data = conv.weight[:, in_channels!=0]
                conv.in_channels = len(in_channels)
            else:
                raise ValueError("len(in_channels)={} must be equal to conv.in_channels={}".format(len(in_channels), conv.in_channels))

        if out_channels is not None:
            if len(out_channels) == conv.out_channels:
                conv.bias.data = conv.bias[out_channels!=0]
                conv.weight.data = conv.weight[out_channels!=0]
                
                print('ConvTranspose2d depth {}/{} = {}'.format(len(in_channels[in_channels!=0]), len(out_channels), len(in_channels[in_channels!=0])/len(out_channels)))
                conv.out_channels = len(out_channels)
            else:
                raise ValueError("len(out_channels)={} must be equal to conv.out_channels={}".format(len(out_channels), conv.out_channels))


    def ApplyStructure(self, msg=None):
        print('ApplyStructure')

        depth = math.ceil(len(self.cells)/2.0)

        print('initial network depth {}/{} = {}'.format(depth, self.unet_depth, depth/self.unet_depth))

        encoder_channel_mask = None
        feedforward_channel_mask = []
        channel_masks = []

        '''_, _, conv_weights = self.ArchitectureWeights()
        newcells = torch.nn.ModuleList()
        for i, conv_weight in enumerate(conv_weights):
            if conv_weight['prune_weight'] < self.feature_threshold:
                print('Prune inactive cell {}'.format(i))
            else:
                newcells.append(self.cells[i])
        self.cells = newcells'''

        # Build up network as if all cells are active
        _, _, conv_weights = self.ArchitectureWeights()
        newcells = torch.nn.ModuleList()
        enc_len = math.floor(len(self.cells)/2.0)
        iDecode = enc_len
        for i in range(enc_len):
            layer_msg = 'Cell {}'.format(i)
            if conv_weights[i]['prune_weight'] < self.feature_threshold:
                prune = True
            else: prune = None
            encoder_channel_mask = self.cells[i].ApplyStructure(encoder_channel_mask, msg=layer_msg, prune=prune)

            feedforward_channel_mask.append(encoder_channel_mask)

        if (len(self.cells) % 2) != 0:
            layer_msg = 'Cell {}'.format(enc_len)
            if conv_weights[enc_len]['prune_weight'] < self.feature_threshold:
                prune = True
            else: prune = None
            encoder_channel_mask = self.cells[enc_len].ApplyStructure(encoder_channel_mask, msg=layer_msg, prune=prune)
            iDecode += 1
        else:
            encoder_channel_mask = torch.zeros_like(encoder_channel_mask, device=self.device) # Only keep feedforward

        for i in range(enc_len):
            iEncDec = i+iDecode
            layer_msg = 'Cell {}'.format(iEncDec)
            if conv_weights[enc_len]['prune_weight'] < self.feature_threshold:
                prune = True
            else: prune = None
            encoder_channel_mask = self.cells[iEncDec].ApplyStructure(encoder_channel_mask, feedforward_channel_mask[-(i+1)], msg=layer_msg, prune=prune)

        return encoder_channel_mask


    def ArchitectureWeights(self):
        architecture_weights = []
        layer_weights = []
        conv_weights = []
        search_structure = []

        for l in self.cells:
            layer_weight, cnn_weight, conv_weight  = l.ArchitectureWeights()
            conv_weights.append(conv_weight)
            architecture_weights.append(layer_weight)

        # Reduce cell weight if it may become inactive as a lower cell approaches 0
        depth = math.floor(len(self.cells)/2.0)
        for i in range(depth):
            prune_weight = []
            prune_weight.append(conv_weights[i]['prune_weight'])
            prune_weight.append(conv_weights[-(i+1)]['prune_weight'])
            if i != 0:
                prune_weight.append(conv_weights[i-1]['prune_weight'])
            prune_weight = torch.min(torch.stack(prune_weight))

            conv_weights[i]['prune_weight'] = prune_weight
            conv_weights[-(i+1)]['prune_weight'] = prune_weight
            architecture_weights[i] *= prune_weight
            architecture_weights[-(i+1)] *= prune_weight

        if len(self.cells) > 2 and (len(self.cells) % 2) != 0:
            prune_weight = []
            prune_weight.append(conv_weights[depth]['prune_weight'])
            prune_weight.append(conv_weights[depth-1]['prune_weight'])
            prune_weight = torch.min(torch.stack(prune_weight))

            conv_weights[depth]['prune_weight'] = prune_weight
            architecture_weights[depth] *= prune_weight

        architecture_weights = torch.cat(architecture_weights)
        architecture_weights = architecture_weights.sum_to_size((1))
            
        return architecture_weights, model_weights(self), conv_weights

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

    parser.add_argument('--debug', '-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-trainingset', type=str, default='data/coco/annotations/instances_train2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-validationset', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-train_image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-imStatistics', type=str2bool, default=False, help='Record individual image statistics')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    #parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')

    parser.add_argument('-dataset', type=str, default='annotations/lit/dataset.yaml', help='Image dataset file')
    parser.add_argument('-class_dict', type=str, default='model/crisplit/lit.json', help='Model class definition file.')

    parser.add_argument('-batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=70, help='Training epochs')
    parser.add_argument('-start_epoch', type=int, default=50, help='Start epoch')
    parser.add_argument('-num_workers', type=int, default=1, help='Training batch size')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='crisplit')
    parser.add_argument('-model_src', type=str,  default='crisplit_20220404h0_01')
    parser.add_argument('-model_dest', type=str, default='crisplit_20220404i0_02')
    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-cuda', type=str2bool, default=True)
    parser.add_argument('-height', type=int, default=640, help='Batch image height')
    parser.add_argument('-width', type=int, default=640, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_GRAYSCALE, help='cv2.imdecode flags')
    parser.add_argument('-learning_rate', type=float, default=1.0e-4, help='Adam learning rate')
    parser.add_argument('-unet_depth', type=int, default=5, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-max_cell_steps', type=int, default=3, help='maximum number of convolution cells in layer to search/minimize')
    parser.add_argument('-channel_multiple', type=float, default=2, help='maximum number of layers to grow per level')
    parser.add_argument('-k_structure', type=float, default=0.1, help='Structure minimization weighting factor')
    parser.add_argument('-k_prune_basis', type=float, default=1.0, help='prune base loss scaling')
    parser.add_argument('-k_prune_exp', type=float, default=50.0, help='prune basis exponential weighting factor')
    parser.add_argument('-k_prune_sigma', type=float, default=0.33, help='prune basis exponential weighting factor')
    parser.add_argument('-target_structure', type=float, default=0.00, help='Structure minimization weighting factor')
    parser.add_argument('-batch_norm', type=str2bool, default=True)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=5.0, help='Channel convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.0, help='cell tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.5, help='convolution channel sigmoid level to prune convolution channels')
    parser.add_argument('-residual', type=str2bool, default=False, help='Residual convolution functions')
    parser.add_argument('-ejector', type=FenceSitterEjectors, default=FenceSitterEjectors.prune_basis, choices=list(FenceSitterEjectors))
    parser.add_argument('-ejector_epoch', type=float, default=0, help='Ejector start epoch')
    parser.add_argument('-ejector_max', type=float, default=1.0, help='Ejector start epoch')
    parser.add_argument('-ejector_exp', type=float, default=3, help='Ejector exponent')
    parser.add_argument('-prune', type=str2bool, default=False)
    parser.add_argument('-train', type=str2bool, default=True)
    parser.add_argument('-infer', type=str2bool, default=False)
    parser.add_argument('-search_structure', type=str2bool, default=True)
    parser.add_argument('-onnx', type=str2bool, default=False)
    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-resultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default='/store/data/network2d')
    parser.add_argument('-tensorboard_dir', type=str, default='./tb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    parser.add_argument('-class_weight', type=json.loads, default='[0.02, 1.0]', help='Loss class weight ')


    parser.add_argument('-description', type=json.loads, default='{"description":"CRISP training"}', help='Test description')

    args = parser.parse_args()
    return args

def MakeNetwork2d(classes, args):
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")

    return Network2d(classes, source_channels=1,
            device=device, 
            unet_depth=args.unet_depth,
            max_cell_steps=args.max_cell_steps, 
            channel_multiple=args.channel_multiple,
            batch_norm=args.batch_norm,
            search_structure=args.search_structure,
            residual=args.residual,
            dropout=args.dropout,
            feature_threshold = args.feature_threshold,
            weight_gain = args.weight_gain,
            convMaskThreshold = args.convMaskThreshold,
            dropout_rate = args.dropout_rate,
            sigmoid_scale = args.sigmoid_scale,
            k_prune_sigma = args.k_prune_sigma)

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    #torch.save(model.state_dict(), out_buffer)
    torch.save(model, out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)

def onnx(model, s3, s3def, args):
    import torch.onnx as torch_onnx

    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, device='cuda')
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

def DisplayImgAn(image, label, segmentation, trainingset, mean, stdev):
    image = np.squeeze(image)
    label = np.squeeze(label)
    segmentation = np.squeeze(segmentation)
    iman = trainingset.store.MergeIman(image, label, mean.item(), stdev.item())
    imseg = trainingset.store.MergeIman(image, segmentation, mean.item(), stdev.item())

    iman = cv2.putText(iman, 'Annotation',(10,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
    imseg = cv2.putText(imseg, 'Segmentation',(10,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
    imanseg = cv2.hconcat([iman, imseg])
    imanseg = cv2.cvtColor(imanseg, cv2.COLOR_BGR2RGB)

    return imanseg

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args): 
    print('Network2D Test')

    test_results={}
    test_results['config'] = args.__dict__
    test_results['config']['ejector'] = args.ejector.value
    test_results['system'] = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'OpenCV': cv2.__version__,
    }
    print('Network2d Test system={}'.format(test_results['system'] ))
    print('Network2d Test config={}'.format(test_results['config'] ))

    torch.autograd.set_detect_anomaly(True)

    s3, creds, s3def = Connect(args.credentails)

    test_results['store'] = s3def

    tb = None
    writer = None
    write_graph = False
    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.class_dict)
    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)

        tb = program.TensorBoard()
        tb.configure(('tensorboard', '--logdir', args.tensorboard_dir))
        url = tb.launch()
        print(f"Tensorboard on {url}")

        writer = SummaryWriter(args.tensorboard_dir)



    # Load dataset
    device = torch.device("cpu")
    pin_memory = False
    if args.cuda:
        device = torch.device("cuda")
        pin_memory = True

    if(args.model_src and args.model_src != ''):
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            segment = torch.load(io.BytesIO(modelObj))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
            return
    else:
        # Create Default segmenter
        segment = MakeNetwork2d(class_dictionary['classes'], args)

    total_parameters = count_parameters(segment)

    if args.prune:
        segment.ApplyParameters(weight_gain=args.weight_gain, 
                                sigmoid_scale=args.sigmoid_scale,
                                feature_threshold=args.feature_threshold,
                                convMaskThreshold=args.convMaskThreshold,
                                k_prune_sigma=args.k_prune_sigma,)
        segment.ApplyStructure()
        reduced_parameters = count_parameters(segment)
        save(segment, s3, s3def, args)
        test_results['prune'] = {'final parameters':reduced_parameters, 'initial parametes' : total_parameters, 'remaning ratio':reduced_parameters/total_parameters }
        print('{} remaining parameters {}/{} = {}'.format(args.model_dest, reduced_parameters, total_parameters, reduced_parameters/total_parameters))

    # Prune with loaded parameters than apply current search_structure setting
    segment.ApplyParameters(search_structure=args.search_structure, convMaskThreshold=args.convMaskThreshold, 
                            k_prune_sigma=args.k_prune_sigma,)
    # Enable multi-gpu processing
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(segment)
        segment = model.module
    else:
        model = segment

    #specify device for model
    segment.to(device)

    # Define a Loss function and optimizer
    target_structure = torch.as_tensor([args.target_structure], dtype=torch.float32, device=device)
    if args.class_weight is not None:
        if len(args.class_weight) == class_dictionary['classes']:
            class_weight = torch.Tensor(args.class_weight).to(device)
        else:
            print('Parameter error: class weight array length={} must equal number of classes.  Exiting'.format(len(args.class_weight, class_dictionary['classes'])))
            return

        if args.cuda:
            class_weight = class_weight.cuda()
    else:
        class_weight = None

    dataset = ImagesDataset(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.class_dict)
    validation_split = .2
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Train
    if args.train:
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(skip_first=10, wait=3, warmup=2, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
                record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:

            loss_fcn = TotalLoss(args.cuda, 
                                  k_structure=args.k_structure, 
                                  target_structure=target_structure, 
                                  class_weight=class_weight, 
                                  search_structure=args.search_structure, 
                                  k_prune_basis=args.k_prune_basis, 
                                  k_prune_exp=args.k_prune_exp,
                                  sigmoid_scale=args.sigmoid_scale,
                                  ejector=args.ejector)
            #optimizer = optim.SGD(segment.parameters(), lr=args.learning_rate, momentum=0.9)
            optimizer = optim.Adam(segment.parameters(), lr= args.learning_rate)
            plotsearch = PlotSearch()
            plotgrads = PlotGradients()
            iSample = 0

            '''trainingset = ImagesDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.trainingset, 
                image_paths=args.train_image_path,
                class_dictionary=class_dictionary, 
                height=args.height, 
                width=args.width, 
                imflags=args.imflags, 
                astype='float32', 
                enable_transform=True)'''

            #dataset = ImagesDataset(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.class_dict)

            #train_batches=int(trainingset.__len__()/args.batch_size)
            #trainloader = torch.utils.data.DataLoader(trainingset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
            train_batches=int(train_indices.__len__()/args.batch_size)
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

            '''testset = ImagesDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
                image_paths=args.val_image_path,
                class_dictionary=class_dictionary, 
                height=args.height, 
                width=args.width, 
                imflags=args.imflags, 
                astype='float32',
                enable_transform=False)'''
            #test_batches=int(testset.__len__()/args.batch_size)
            #testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
            test_batches=int(val_indices.__len__()/args.batch_size)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=pin_memory, collate_fn=collate_fn)
            test_freq = 10*int(math.ceil(train_batches/test_batches))
            tstart = None
            batch_per_epoch = int(len(train_indices)/args.batch_size)
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            # Set up fence sitter ejectors
            ejector_exp = None
            if args.ejector == FenceSitterEjectors.dais or args.ejector == FenceSitterEjectors.dais.value:
                writer.add_scalar('CRISP/sigmoid_scale', args.sigmoid_scale, iSample)
                if args.epochs > args.ejector_epoch and args.ejector_max > args.sigmoid_scale:
                    ejector_exp =  Exponential(vx=args.ejector_epoch, vy=args.sigmoid_scale, px=args.ejector_epoch+args.epochs, py=args.ejector_max, power=args.ejector_exp)

            elif args.ejector == FenceSitterEjectors.prune_basis or args.ejector == FenceSitterEjectors.prune_basis.value:
                writer.add_scalar('CRISP/k_prune_basis', args.k_prune_basis, iSample)
                if args.epochs > args.ejector_epoch and args.ejector_max > 0:
                    ejector_exp =  Exponential(vx=args.ejector_epoch, vy=0, px=args.ejector_epoch+args.epochs, py=args.ejector_max, power=args.ejector_exp)


            for epoch in tqdm(range(args.start_epoch, args.epochs), bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}', desc="Epochs", disable=args.job):  # loop over the dataset multiple times
                iTest = iter(testloader)

                running_loss = 0.0
                for i, data in tqdm(enumerate(trainloader), bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}', total=batch_per_epoch, desc="Batches", disable=args.job):
                    # get the inputs; data is a list of [inputs, labels]
                    prevtstart = tstart
                    tstart = time.perf_counter()

                    inputs, labels, mean, stdev = data

                    if args.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad(set_to_none=True)

                    #with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    tinfer = time.perf_counter()
                    loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale = loss_fcn(outputs, labels, model)
                    tloss = time.perf_counter()
                    loss.backward()
                    optimizer.step()
                    tend = time.perf_counter()

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
                        writer.add_scalar('loss/train', loss, iSample)
                        writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, iSample)
                        writer.add_scalar('time/infer', dtInfer, iSample)
                        writer.add_scalar('time/loss', dtLoss, iSample)
                        writer.add_scalar('time/backpropegation', dtBackprop, iSample)
                        writer.add_scalar('time/compute', dtCompute, iSample)
                        writer.add_scalar('time/cycle', dtCycle, iSample)
                        writer.add_scalar('CRISP/architecture_loss', architecture_loss, iSample)
                        writer.add_scalar('CRISP/prune_loss', prune_loss, iSample)
                        writer.add_scalar('CRISP/architecture_reduction', architecture_reduction, iSample)
                        #writer.add_scalar('CRISP/sigmoid_scale', sigmoid_scale, iSample)

                    if i % test_freq == test_freq-1:    # Save image and run test
                        if writer is not None:
                            imprune_weights = plotsearch.plot(cell_weights)
                            if imprune_weights.size > 0:
                                im_class_weights = cv2.cvtColor(imprune_weights, cv2.COLOR_BGR2RGB)
                                writer.add_image('network/prune_weights', im_class_weights, 0,dataformats='HWC')

                            imgrad = plotgrads.plot(segment)
                            if imgrad.size > 0:
                                im_grad_norm = cv2.cvtColor(imgrad, cv2.COLOR_BGR2RGB)
                                writer.add_image('network/gradient_norm', im_grad_norm, 0,dataformats='HWC')

                        images = inputs.cpu().permute(0, 2, 3, 1).numpy()
                        labels = np.around(labels.cpu().numpy()).astype('uint8')
                        segmentations = torch.argmax(outputs, 1)
                        segmentations = segmentations.cpu().numpy().astype('uint8')
                        if writer is not None:
                            if not write_graph:
                                writer.add_graph(model, inputs)
                                write_graph = True
                            for j in range(1):
                                imanseg = DisplayImgAn(images[j], labels[j], segmentations[j], dataset, mean[j], stdev[j])      
                                writer.add_image('segmentation/train', imanseg, 0,dataformats='HWC')

                        with torch.no_grad():
                            data = next(iTest)
                            inputs, labels, mean, stdev = data
                            if args.cuda:
                                inputs = inputs.cuda()
                                labels = labels.cuda()

                            #with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss, cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale = loss_fcn(outputs, labels, model)

                        if writer is not None:
                            writer.add_scalar('loss/test', loss, iSample)
                            writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, iSample)

                            

                        running_loss /=test_freq
                        msg = '[{:3}/{}, {:6d}/{}]  loss: {:0.5e}|{:0.5e} cross-entropy loss: {:0.5e}|{:0.5e} remaining: {:0.5e} (train|test) step time: {:0.3f}'.format(
                            epoch + 1, args.epochs, i + 1, batch_per_epoch, 
                            running_loss, loss.item(),
                            training_cross_entropy_loss.item(), cross_entropy_loss.item(), 
                            architecture_reduction.item(), dtCycle)
                        if args.job is True:
                            print(msg)
                        else:
                            tqdm.write(msg)
                        running_loss = 0.0

                        images = inputs.cpu().permute(0, 2, 3, 1).numpy()
                        labels = np.around(labels.cpu().numpy()).astype('uint8')
                        segmentations = torch.argmax(outputs, 1)
                        segmentations = segmentations.cpu().numpy().astype('uint8')

                        if writer is not None:
                            for j in range(1):
                                imanseg = DisplayImgAn(images[j], labels[j], segmentations[j], dataset, mean[j], stdev[j])      
                                writer.add_image('segmentation/test', imanseg, 0,dataformats='HWC')

                    iSave = 1000
                    if i % iSave == iSave-1:    # print every iSave mini-batches
                        img = plotsearch.plot(cell_weights)
                        if img.size > 0:
                            is_success, buffer = cv2.imencode(".png", img, compression_params)
                            img_enc = io.BytesIO(buffer).read()
                            filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                            s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)
                        imgrad = plotgrads.plot(segment)
                        if imgrad.size > 0:
                            is_success, buffer = cv2.imencode(".png", imgrad)  
                            img_enc = io.BytesIO(buffer).read()
                            filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                            s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)
                            # Save calls zero_grads so call it after plotgrads.plot

                        save(segment, s3, s3def, args)

                    if args.fast and i+1 >= test_freq:
                        break
                
                    prof.step()
                    iSample += 1

                img = plotsearch.plot(cell_weights)
                if img.size > 0:
                    is_success, buffer = cv2.imencode(".png", img, compression_params)
                    img_enc = io.BytesIO(buffer).read()
                    filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                    s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

                # Plot gradients before saving which clears the gradients
                imgrad = plotgrads.plot(segment)
                if imgrad.size > 0:
                    is_success, buffer = cv2.imencode(".png", imgrad)  
                    img_enc = io.BytesIO(buffer).read()
                    filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                    s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

                save(segment, s3, s3def, args)

                if ejector_exp is not None and (args.ejector == FenceSitterEjectors.dais or args.ejector == FenceSitterEjectors.dais.value):
                    sigmoid_scale = ejector_exp.f(epoch)
                    segment.ApplyParameters(sigmoid_scale=sigmoid_scale)
                    writer.add_scalar('CRISP/sigmoid_scale', sigmoid_scale, iSample)
                elif args.ejector == FenceSitterEjectors.prune_basis or args.ejector == FenceSitterEjectors.prune_basis.value:
                    k_prune_basis = ejector_exp.f(epoch)
                    loss_fcn.k_prune_basis = k_prune_basis
                    writer.add_scalar('CRISP/k_prune_basis', k_prune_basis, iSample)

            print('{} training complete'.format(args.model_dest))
            test_results['training'] = {}
            if cross_entropy_loss: test_results['training']['cross_entropy_loss']=cross_entropy_loss.item()
            if architecture_loss: test_results['training']['architecture_loss']=architecture_loss.item()
            if prune_loss: test_results['training']['prune_loss']=prune_loss.item()
            if architecture_reduction: test_results['training']['architecture_reduction']=architecture_reduction.item()

            if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.model_dest is not None and len(args.model_dest) > 0):
                tb_path = '{}/{}/{}_tb'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
                s3.PutDir(s3def['sets']['test']['bucket'], args.tensorboard_dir, tb_path )

    if args.infer:

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        test_summary = {'date':date_time}

        '''testset = ImagesDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
            image_paths=args.val_image_path,
            class_dictionary=class_dictionary, 
            height=args.height, 
            width=args.width, 
            imflags=args.imflags,
            astype='float32',
            enable_transform=False)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, pin_memory=pin_memory)'''
        test_batches=int(val_indices.__len__()/args.batch_size)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=pin_memory, collate_fn=collate_fn)

        if args.test_dir is not None:
            outputdir = '{}/{}'.format(args.test_dir,args.model_class)
            os.makedirs(outputdir, exist_ok=True)
        else:
            outputdir = None

        dsResults = DatasetResults(class_dictionary, args.batch_size, imStatistics=args.imStatistics, imgSave=outputdir)

        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=20, warmup=12, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
                record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:

            for i, data in tqdm(enumerate(testloader), total=test_batches, desc="Inference steps", disable=args.job):
                images, labels, mean, stdev = data
                if args.cuda:
                    images = images.cuda()

                initial = datetime.now()

                outputs = segment(images)
                segmentations = torch.argmax(outputs, 1)
                dt = (datetime.now()-initial).total_seconds()
                imageTime = dt/args.batch_size

                images = images.cpu().permute(0, 2, 3, 1).numpy()
                labels = np.around(labels.cpu().numpy()).astype('uint8')
                segmentations = segmentations.cpu().numpy().astype('uint8')

                dsResults.infer_results(i, images, labels, segmentations, mean.numpy(), stdev.numpy(), dt)

                if args.fast and i+1 >= 10:
                    break
                prof.step() 

        test_summary['objects'] = dsResults.objTypes
        test_summary['object store'] =s3def
        test_summary['results'] = dsResults.Results()
        test_summary['config'] = args.__dict__
        if args.ejector is not None and type(args.ejector) != str:
            test_summary['config']['ejector'] = args.ejector.value
        test_summary['system'] = test_results['system']

        # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
        test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
        training_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)
        if training_data is None or type(training_data) is not list:
            training_data = []
        training_data.append(test_summary)
        s3.PutDict(s3def['sets']['test']['bucket'], test_path, training_data)

        test_url = s3.GetUrl(s3def['sets']['test']['bucket'], test_path)
        print("Test results {}".format(test_url))

        test_results['test'] = test_summary['results']

    if args.onnx:
        onnx(segment, s3, s3def, args)

    if args.resultspath is not None and len(args.resultspath) > 0:
        WriteDict(test_results, args.resultspath)

    print('Finished {} {}'.format(args.model_dest, test_results))
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

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client()
        print("Debugger attached")

    result = Test(args)
    sys.exit(result)

