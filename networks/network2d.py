#!/usr/bin/python3
import math
import os
import sys
import copy
import io
import json
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
import cv2
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.abspath(''))
from networks.cell2d import Cell, NormGausBasis
from utils.torch_util import count_parameters, model_stats, model_weights
from utils.jsonutil import ReadDictJson
from utils.s3 import s3store, Connect
from datasets.cocostore import CocoDataset
from utils.metrics import similarity, jaccard, DatasetResults
from networks.totalloss import TotalLoss


DefaultMaxDepth = 4
class Network2d(nn.Module):
    def __init__(self, 
                 out_channels=1, 
                 source_channels=3, 
                 initial_channels=64, 
                 is_cuda=True, 
                 min_search_depth=3, 
                 max_search_depth=DefaultMaxDepth, 
                 max_cell_steps=6, 
                 channel_multiple=2, 
                 batch_norm=True, 
                 cell=Cell, 
                 search_structure=True,
                 residual = False,
                 depth = float(DefaultMaxDepth),
                 feature_threshold=0.5,
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 dropout_rate = 0.2,
                 sigmoid_scale = 5.0):
        super(Network2d, self).__init__()

        self.max_search_depth = max_search_depth
        self.min_search_depth = min_search_depth
        self.out_channels = out_channels
        self.source_channels = source_channels
        self.initial_channels = initial_channels
        self.is_cuda = is_cuda
        self.cell = cell
        self.max_cell_steps = max_cell_steps
        self.channel_multiple = channel_multiple
        self.depth = torch.nn.Parameter(torch.tensor(depth, dtype=torch.float)) # Initial depth parameter = max_search_depth
        self.batch_norm = batch_norm

        self.encode_decode = torch.nn.ModuleList()
        self.upsample = torch.nn.ModuleList()
        self.final_conv = torch.nn.ModuleList()
        self.search_structure = search_structure
        self.residual = residual

        self.feature_threshold=feature_threshold
        self.weight_gain = weight_gain
        self.convMaskThreshold=convMaskThreshold
        self.dropout_rate = dropout_rate
        self.sigmoid_scale = sigmoid_scale

        encoder_channels = self.initial_channels
        prev_encoder_chanels = self.source_channels
        feedforward_chanels = []

        convolutions=[{'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':self.search_structure},
                        {'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':self.search_structure}]
        for i in range(self.max_search_depth-1):
            for convoluiton in convolutions:
                convoluiton['out_channels'] = encoder_channels

            cell = self.cell(prev_encoder_chanels, 
                             batch_norm=self.batch_norm, 
                             is_cuda=self.is_cuda,
                             convolutions=convolutions,
                             search_structure=self.search_structure,
                             residual=self.residual,
                             dropout_rate=self.dropout_rate, 
                             sigmoid_scale=self.sigmoid_scale, 
                             feature_threshold=self.feature_threshold)
            self.encode_decode.append(cell)

            feedforward_chanels.append(encoder_channels)
            prev_encoder_chanels = encoder_channels
            encoder_channels = int(self.channel_multiple*encoder_channels)

        for convoluiton in convolutions:
            convoluiton['out_channels'] = encoder_channels

        cell = self.cell(prev_encoder_chanels,
                         batch_norm=self.batch_norm,
                         is_cuda=self.is_cuda,
                         convolutions=convolutions,
                         search_structure=self.search_structure,
                         residual=self.residual,
                         dropout_rate=self.dropout_rate, 
                         sigmoid_scale=self.sigmoid_scale, 
                         feature_threshold=self.feature_threshold)
        self.encode_decode.append(cell)

        for i in range(self.max_search_depth-2):

            #if definition is not None and len(definition['encode_decode']) > i:
            #    encoder_channels = definition['encode_decode'][self.max_search_depth+i]['in1_channels']
            #    self.upsample.append(nn.ConvTranspose2d(encoder_channels, encoder_channels, 2, stride=2))
            #    cell = self.cell(definition=definition['encode_decode'][self.max_search_depth+i])
            #else:
            self.upsample.append(nn.ConvTranspose2d(encoder_channels, encoder_channels, 2, stride=2))
            prev_encoder_chanels = encoder_channels
            encoder_channels = int(encoder_channels/self.channel_multiple)

            convolutions=[{'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':self.search_structure},
                          {'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':self.search_structure},
                          {'out_channels':int(encoder_channels/2), 'kernel_size': 2, 'stride': 2, 'dilation': 1, 'search_structure':self.search_structure}]

            cell = self.cell(prev_encoder_chanels, 
                             feedforward_chanels[-(i+1)],
                             batch_norm=self.batch_norm,
                             is_cuda=self.is_cuda,
                             convolutions=convolutions,
                             search_structure=self.search_structure,
                             residual=self.residual,
                             dropout_rate=self.dropout_rate, 
                             sigmoid_scale=self.sigmoid_scale, 
                             feature_threshold=self.feature_threshold)
            self.encode_decode.append(cell)

        prev_encoder_chanels = encoder_channels
        convolutions=[{'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True},
                      {'out_channels':encoder_channels, 'kernel_size': 3, 'stride': 1, 'dilation': 1, 'search_structure':True},
                      {'out_channels':self.out_channels, 'kernel_size': 1, 'stride': 1, 'dilation': 1, 'search_structure':True}]

        cell = self.cell(prev_encoder_chanels,
                         feedforward_chanels[0],
                         is_cuda=self.is_cuda,
                         batch_norm=self.batch_norm,
                         convolutions=convolutions,
                         search_structure=self.search_structure,
                         residual=self.residual,
                         dropout_rate=self.dropout_rate, 
                         sigmoid_scale=self.sigmoid_scale, 
                         feature_threshold=self.feature_threshold)
        self.encode_decode.append(cell)

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def definition(self):
        definition_dict = {
            'out_channels': self.out_channels,
            'source_channels': self.source_channels,
            'initial_channels': self.initial_channels,
            'is_cuda': self.is_cuda,
            'min_search_depth': self.min_search_depth,
            'max_search_depth': self.max_search_depth,
            'max_cell_steps': self.max_cell_steps,
            'channel_multiple': self.channel_multiple,
            'batch_norm': self.batch_norm,
            'search_structure': self.search_structure,
            'depth': self.depth.item(),
            'encode_decode': [],
        }

        for ed in self.encode_decode:
            definition_dict['encode_decode'].append(ed.definition())

        architecture_weights, total_trainable_weights = self.ArchitectureWeights()
        definition_dict['architecture_weights']= architecture_weights.item()
        definition_dict['total_trainable_weights']= total_trainable_weights.item()

        return definition_dict

    def forward_depth(self, x, depth):
        feed_forward = []
        for i in range(depth-1):
            x = self.encode_decode[i](x)
            feed_forward.append(x)
            x = self.pool(x)

        # Feed-through
        x = self.encode_decode[depth-1](x)
        decode_depth = len(self.encode_decode)-depth
        upsaple_depth = self.max_search_depth-depth
        for i in range(depth-1):
            x = self.upsample[upsaple_depth+i](x)
            x = self.encode_decode[decode_depth+i](x, feed_forward[-(i+1)])

        x = self.encode_decode[-1](x) # Size to output
        return x

    def forward(self, x):

        # feed_forward = []
        # for i in range(self.max_search_depth-1):
        #     x = self.encode_decode[i](x)
        #     feed_forward.append(x)
        #     x = self.pool(x)
        #     x = x*self.sigmoid(self.depth-i)


        # # Feed-through
        # x = self.encode_decode[self.max_search_depth-1](x)
        # x = x*self.sigmoid(self.depth-i)

        # for i in range(self.max_search_depth-1):
        #     x = self.upsample[i](x)
        #     x = self.encode_decode[i+self.max_search_depth](x, feed_forward[-(i+1)])
        #     x = x*self.sigmoid(self.depth-i)

        # x = self.encode_decode[-1](x) # Size to output

        if self.search_structure:
            y = None
            search_range = self.max_search_depth-self.min_search_depth+1
            for i in range(search_range):
                xi = self.forward_depth(x, self.min_search_depth+i)
                xi = xi*NormGausBasis(search_range, i, self.depth-self.min_search_depth)
                if y is None:
                    y=xi
                else:
                    y = y+xi
        else:
            y = self.forward_depth(x, self.max_search_depth)

        return y

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


    def ApplyStructure(self):
        print('ApplyStructure')


        depth = round(self.depth.item())
        new_depth = depth
        if depth <=self.min_search_depth:
            new_depth = self.min_search_depth
        elif depth < self.max_search_depth:
            new_depth = depth
        else:
            new_depth = self.max_search_depth
        print('network depth {}/{} = {}'.format(new_depth, self.max_search_depth, new_depth/self.max_search_depth))

        encoder_channel_mask = None
        feedforward_channel_mask = []
        new_encode_decode = torch.nn.ModuleList()
        new_upsample = torch.nn.ModuleList()
        
        for i in range(new_depth-1):
            encoder_channel_mask = self.encode_decode[i].ApplyStructure(encoder_channel_mask)
            new_encode_decode.append(self.encode_decode[i])
            feedforward_channel_mask.append(encoder_channel_mask)

        encoder_channel_mask = self.encode_decode[new_depth-1].ApplyStructure(encoder_channel_mask)
        new_encode_decode.append(self.encode_decode[new_depth-1])

        for i in range(new_depth-1):
            iUpsample = i+self.max_search_depth-new_depth
            self.ApplyStructureConvTranspose2d(self.upsample[iUpsample], encoder_channel_mask, encoder_channel_mask) # Remove same input and output channels
            new_upsample.append(self.upsample[iUpsample])
            iEncDec = i+2*self.max_search_depth-new_depth
            encoder_channel_mask = self.encode_decode[iEncDec].ApplyStructure(encoder_channel_mask, feedforward_channel_mask[-(i+1)])
            new_encode_decode.append(self.encode_decode[iEncDec])

        encoder_channel_mask = self.encode_decode[-1].ApplyStructure(encoder_channel_mask) # Final resize to output features
        new_encode_decode.append(self.encode_decode[-1])

        self.encode_decode = new_encode_decode
        self.upsample = new_upsample
        self.max_search_depth = new_depth

        return encoder_channel_mask


    def ArchitectureWeights(self):
        #print('ArchitectureWeights')

        architecture_weights = torch.zeros(1)
        total_trainable_weights = torch.zeros(1)
        torch_total_trainable_weights = torch.tensor(model_weights(self))


        if self.is_cuda:
            architecture_weights = architecture_weights.cuda()
            total_trainable_weights = total_trainable_weights.cuda()

        for j in range(self.max_search_depth-1):
            
            encode_architecture_weights, step_total_trainable_weights = self.encode_decode[j].ArchitectureWeights()
            total_trainable_weights += step_total_trainable_weights
            decode_architecture_weights, step_total_trainable_weights = self.encode_decode[-(j+2)].ArchitectureWeights()
            total_trainable_weights += step_total_trainable_weights
            #total_trainable_weights += torch.tensor(model_weights(self.upsample[j])) # Don't include until accounting is correct

            # Sigmoid weightingof architecture weighting to prune model depth
            layer_weights = encode_architecture_weights+decode_architecture_weights
            #architecture_weights += layer_weights
            depth_weighted_architecture_weight = layer_weights*self.sigmoid(self.depth-j)
            architecture_weights += depth_weighted_architecture_weight


        encode_architecture_weights, step_total_trainable_weights = self.encode_decode[self.max_search_depth].ArchitectureWeights()
        architecture_weights += encode_architecture_weights
        total_trainable_weights += step_total_trainable_weights

        encode_architecture_weights, step_total_trainable_weights = self.encode_decode[-1].ArchitectureWeights()
        architecture_weights += encode_architecture_weights
        total_trainable_weights += step_total_trainable_weights

        return architecture_weights, total_trainable_weights


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
    parser.add_argument('-imStatistics', type=bool, default=False, help='Record individual image statistics')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')

    parser.add_argument('-batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='segmin')
    parser.add_argument('-model_src', type=str,  default=None)
    parser.add_argument('-model_dest', type=str, default='segment_nas_512x442_20220211_00')
    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-height', type=int, default=480, help='Batch image height')
    parser.add_argument('-width', type=int, default=512, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')
    parser.add_argument('-learning_rate', type=float, default=1.0e-4, help='Adam learning rate')
    parser.add_argument('-max_search_depth', type=int, default=5, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-min_search_depth', type=int, default=2, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-max_cell_steps', type=int, default=3, help='maximum number of convolution cells in layer to search/minimize')
    parser.add_argument('-channel_multiple', type=float, default=2, help='maximum number of layers to grow per level')
    parser.add_argument('-k_structure', type=float, default=1.0e-4, help='Structure minimization weighting fator')
    parser.add_argument('-target_structure', type=float, default=0.01, help='Structure minimization weighting fator')
    parser.add_argument('-batch_norm', type=bool, default=False)
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.5, help='tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.5, help='sigmoid level to prune convolution chanmels')
    parser.add_argument('-residual', type=bool, default=False, help='Residual convolution functionss')

    parser.add_argument('-prune', type=bool, default=False)
    parser.add_argument('-train', type=bool, default=False)
    parser.add_argument('-infer', type=bool, default=True)
    parser.add_argument('-search_structure', type=bool, default=True)
    parser.add_argument('-onnx', type=bool, default=False)

    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='/store/test/nassegtb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir /store/test/nassegtb --bind_all')
    parser.add_argument('-class_weight', type=json.loads, default='[1.0,1.0, 1.0, 1.0]', help='Loss class weight ') 

    parser.add_argument('-description', type=json.loads, default='{"description":"NAS segmentation"}', help='Test description')

    args = parser.parse_args()
    return args

def MakeNetwork2d(classes, args):
    return Network2d(classes, 
            is_cuda=args.cuda, 
            max_search_depth=args.max_search_depth,
            min_search_depth=args.min_search_depth, 
            max_cell_steps=args.max_cell_steps, 
            channel_multiple=args.channel_multiple,
            batch_norm=args.batch_norm,
            search_structure=args.search_structure,
            residual=args.residual,
            depth = args.max_search_depth,
            feature_threshold = args.feature_threshold,
            weight_gain = args.weight_gain,
            convMaskThreshold = args.convMaskThreshold,
            dropout_rate = args.dropout_rate,
            sigmoid_scale = args.sigmoid_scale)

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model.state_dict(), out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)
    s3.PutDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), model.definition())

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

def InferLoss(inputs, labels, args, model, criterion, optimizer):
    if args.cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # zero the parameter gradients
    model.zero_grad(set_to_none=True)

    #with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, model)

    return outputs, loss, cross_entropy_loss, architecture_loss

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    print('Network2D Test')

    system = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'numpy version': sys.modules['numpy'].__version__,
    }

    print('system={}'.format(system))

    torch.autograd.set_detect_anomaly(True)

    s3, creds, s3def = Connect(args.credentails)

    class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.class_dict)
    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)

    # Load dataset
    device = "cpu"
    pin_memory = False
    if args.cuda:
        device = "cuda"
        pin_memory = True

    test_freq = 20
    if args.train:
        trainingset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.trainingset, 
            image_paths=args.train_image_path,
            class_dictionary=class_dictionary, 
            height=args.height, 
            width=args.width, 
            imflags=args.imflags, 
            astype='float32', 
            enable_transform=True)

        train_batches=int(trainingset.__len__()/args.batch_size)
        trainloader = torch.utils.data.DataLoader(trainingset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

        testset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
            image_paths=args.val_image_path,
            class_dictionary=class_dictionary, 
            height=args.height, 
            width=args.width, 
            imflags=args.imflags, 
            astype='float32',
            enable_transform=False)
        test_batches=int(testset.__len__()/args.batch_size)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
        test_freq = int(math.ceil(train_batches/test_batches))

    if(args.model_src and args.model_src != ''):
        #modeldict = s3.GetDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        segment = MakeNetwork2d(class_dictionary['classes'], args)

        if modelObj is not None:
            # segment.load_state_dict(torch.load(io.BytesIO(modelObj)))
            segment = torch.load(io.BytesIO(modelObj))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
            return
    else:
        # Create Default segmenter
        segment = MakeNetwork2d(class_dictionary['classes'], args)

    #sepcificy device for model
    segment.to(device)

    total_parameters = count_parameters(segment)

    if args.prune:
        segment.ApplyStructure()
        reduced_parameters = count_parameters(segment)
        save(segment, s3, s3def, args)
        print('Reduced parameters {}/{} = {}'.format(reduced_parameters, total_parameters, reduced_parameters/total_parameters))

    # Define a Loss function and optimizer
    target_structure = torch.as_tensor([args.target_structure], dtype=torch.float32)
    if args.class_weight is not None:
        if len(args.class_weight) == class_dictionary['classes']:
            class_weight = torch.Tensor(args.class_weight)
        else:
            print('Parameter error: class weight array length={} must equal number of classes.  Exiting'.format(len(args.class_weight, class_dictionary['classes'])))
            return

        if args.cuda:
            class_weight = class_weight.cuda()
    else:
        class_weight = None

    if args.cuda:
        target_structure = target_structure.cuda()
        class_weight = class_weight.cuda()
    criterion = TotalLoss(args.cuda, k_structure=args.k_structure, target_structure=target_structure, class_weight=class_weight, search_structure=args.search_structure)
    #optimizer = optim.SGD(segment.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(segment.parameters(), lr= args.learning_rate)

    # Train
    if args.train:
        for epoch in tqdm(range(args.epochs), desc="Train epochs"):  # loop over the dataset multiple times
            iTest = iter(testloader)

            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader), total=trainingset.__len__()/args.batch_size, desc="Train steps"):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, mean, stdev = data

                outputs, loss, cross_entropy_loss, architecture_loss = InferLoss(inputs, labels, args, segment, criterion, optimizer)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                writer.add_scalar('loss/train', loss, i)
                writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, i)
                writer.add_scalar('architecture_loss/train', architecture_loss, i)

                if i % test_freq == test_freq-1:    # Save image and run test

                    data = next(iTest)
                    inputs, labels, mean, stdev = data
                    outputs, loss, cross_entropy_loss, architecture_loss = InferLoss(inputs, labels, args, segment, criterion, optimizer)

                    writer.add_scalar('loss/test', loss, int((i+1)/test_freq-1))
                    writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, int((i+1)/test_freq-1))
                    writer.add_scalar('architecture_loss/test', architecture_loss, int((i+1)/test_freq-1))

                    running_loss /=test_freq
                    tqdm.write('Train [{}, {:06}] loss: {:0.5e}'.format(epoch + 1, i + 1, running_loss))
                    running_loss = 0.0

                    images = inputs.cpu().permute(0, 2, 3, 1).numpy()
                    labels = np.around(labels.cpu().numpy()).astype('uint8')
                    segmentations = torch.argmax(outputs, 1)
                    segmentations = segmentations.cpu().numpy().astype('uint8')

                    for j in range(1):
                        image = np.squeeze(images[j])
                        label = np.squeeze(labels[j])
                        segmentation = np.squeeze(segmentations[j])
                        iman = trainingset.coco.MergeIman(image, label, mean[j].item(), stdev[j].item())
                        imseg = trainingset.coco.MergeIman(image, segmentation, mean[j].item(), stdev[j].item())

                        iman = cv2.putText(iman, 'Annotation',(10,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
                        imseg = cv2.putText(imseg, 'Segmentation',(10,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
                        imanseg = cv2.hconcat([iman, imseg])
                        imanseg = cv2.cvtColor(imanseg, cv2.COLOR_BGR2RGB)

                        writer.add_image('segment', imanseg, 0,dataformats='HWC')

                iSave = 2000
                if i % iSave == iSave-1:    # print every 20 mini-batches
                    save(segment, s3, s3def, args)

                if args.fast and i+1 >= test_freq:
                    break

            save(segment, s3, s3def, args)


    if args.infer:

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        test_summary = {'date':date_time}

        testset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
            image_paths=args.val_image_path,
            class_dictionary=class_dictionary, 
            height=args.height, 
            width=args.width, 
            imflags=args.imflags,
            astype='float32',
            enable_transform=False)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

        if args.test_dir is not None:
            outputdir = '{}/{}'.format(args.test_dir,args.model_class)
            os.makedirs(outputdir, exist_ok=True)
        else:
            outputdir = None

        dsResults = DatasetResults(class_dictionary, args.batch_size, imStatistics=args.imStatistics, imgSave=outputdir)

        for i, data in tqdm(enumerate(testloader), total=int(testset.__len__()/args.batch_size), desc="Inference steps"):
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

            if args.fast and i+1 >= test_freq:
                break

        test_summary['objects'] = dsResults.objTypes
        test_summary['object store'] =s3def
        test_summary['results'] = dsResults.Results()
        test_summary['config'] = args.__dict__
        test_summary['system'] = system

        # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
        test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
        training_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)
        if training_data is None or type(training_data) is not list:
            training_data = []
        training_data.append(test_summary)
        s3.PutDict(s3def['sets']['test']['bucket'], test_path, training_data)

        test_url = s3.GetUrl(s3def['sets']['test']['bucket'], test_path)
        print("Test results {}".format(test_url))

    if args.onnx:
        onnx(segment, s3, s3def, args)

    print('Finished network2d Test')
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
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    result = Test(args)
    sys.exit(result)

