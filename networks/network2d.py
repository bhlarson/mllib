import math
import os
import sys
import copy
import io
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
import cv2
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath(''))
from networks.cell2d import Cell, NormGausBasis
from utils.torch_util import count_parameters, model_stats, model_weights
from utils.jsonutil import ReadDictJson
from utils.s3 import s3store, Connect
from datasets.cocostore import CocoDataset
from utils.metrics import similarity, jaccard, DatasetResults
#from utils.similarity import similarity, jaccard
# from segment.display import DrawFeatures
from torch.utils.tensorboard import SummaryWriter

DefaultMaxDepth = 7
class Network2d(nn.Module):
    def __init__(self, 
                 out_channels=1, 
                 source_channels=3, 
                 initial_channels=64, 
                 is_cuda=True, 
                 min_search_depth=3, 
                 max_search_depth=DefaultMaxDepth, 
                 max_cell_steps=6, 
                 channel_multiple=1.5, 
                 batch_norm=True, 
                 cell=Cell, 
                 search_structure=True,
                 depth = float(DefaultMaxDepth),
                 definition=None):
        super(Network2d, self).__init__()

        if definition is not None:
            out_channels = definition['out_channels']
            source_channels = definition['source_channels']
            initial_channels = definition['initial_channels']
            is_cuda = definition['is_cuda']
            min_search_depth = definition['min_search_depth']
            max_search_depth = definition['max_search_depth']
            max_cell_steps = definition['max_cell_steps']
            channel_multiple = definition['channel_multiple']
            batch_norm = definition['batch_norm']
            search_structure = definition['search_structure']
            depth = definition['depth']


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

        encoder_channels = self.initial_channels
        prev_encoder_chanels = self.source_channels
        feedforward_chanels = []

        for i in range(self.max_search_depth-1):
            if definition is not None and len(definition['encode_decode']) > i:
                cell = self.cell(definition=definition['encode_decode'][i])
            else:
                cell = self.cell(self.max_cell_steps, 
                             encoder_channels, 
                             prev_encoder_chanels, 
                             is_cuda=self.is_cuda, 
                             batch_norm=self.batch_norm, 
                             search_structure=self.search_structure, 
                             depth=self.max_cell_steps)
            self.encode_decode.append(cell)

            feedforward_chanels.append(encoder_channels)
            prev_encoder_chanels = encoder_channels
            encoder_channels = int(self.channel_multiple*encoder_channels)

        if definition is not None and len(definition['encode_decode']) > i:
            cell = self.cell(definition=definition['encode_decode'][self.max_search_depth-1])
        else:
            cell = self.cell(self.max_cell_steps, 
                            encoder_channels, 
                            prev_encoder_chanels, 
                            is_cuda=self.is_cuda, 
                            batch_norm=self.batch_norm, 
                            search_structure=self.search_structure,
                            depth=self.max_cell_steps)
        self.encode_decode.append(cell)

        for i in range(self.max_search_depth-1):

            if definition is not None and len(definition['encode_decode']) > i:
                encoder_channels = definition['encode_decode'][self.max_search_depth+i]['in1_channels']
                self.upsample.append(nn.ConvTranspose2d(encoder_channels, encoder_channels, 2, stride=2))
                cell = self.cell(definition=definition['encode_decode'][self.max_search_depth+i])
            else:
                self.upsample.append(nn.ConvTranspose2d(encoder_channels, encoder_channels, 2, stride=2))
                prev_encoder_chanels = encoder_channels
                encoder_channels = int(encoder_channels/self.channel_multiple)
                cell = self.cell(self.max_cell_steps, 
                                encoder_channels, 
                                prev_encoder_chanels, 
                                feedforward_chanels[-(i+1)], 
                                is_cuda=self.is_cuda, 
                                batch_norm=self.batch_norm, 
                                search_structure=self.search_structure,
                                depth=self.max_cell_steps)
            self.encode_decode.append(cell)       

        if definition is not None and len(definition['encode_decode']) > i:
            cell = self.cell(definition=definition['encode_decode'][-1])
        else:
            cell = self.cell(self.max_cell_steps, 
                            out_channels, 
                            encoder_channels, 
                            is_cuda=self.is_cuda, 
                            batch_norm=self.batch_norm, 
                            search_structure=self.search_structure,
                            depth=self.max_cell_steps)
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

class TotalLoss(torch.nn.modules.loss._WeightedLoss):
    # https://github.com/JunMa11/SegLoss
    """This criterion combines :class:`~torch.nn.LogSoftmax` and :class:`~torch.nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch. If the
    :attr:`weight` argument is specified then this is a weighted average:

    .. math::
        \text{loss} = \frac{\sum^{N}_{i=1} loss(i, class[i])}{\sum^{N}_{i=1} weight[class[i]]}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`prune` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        prune (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`prune` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`prune` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 prune=None, reduction: str = 'mean', k_structure=0.0, target_structure=torch.as_tensor([1.0], dtype=torch.float32), class_weight=None, search_structure=True) -> None:
        super(TotalLoss, self).__init__(weight, size_average, prune, reduction)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.k_structure = k_structure
        self.softsign = nn.Softsign()
        self.target_structure = target_structure  
        self.mseloss = nn.MSELoss()
        self.class_weight = class_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        self.search_structure = search_structure


    def forward(self, input: torch.Tensor, target: torch.Tensor, network) -> torch.Tensor:
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        loss = self.cross_entropy_loss(input, target.long())

        dims = []
        depths = []
        architecture_weights, total_trainable_weights = network.ArchitectureWeights()
        arcitecture_reduction = architecture_weights/total_trainable_weights
        architecture_loss = self.mseloss(arcitecture_reduction,self.target_structure)

        total_loss = loss
        if self.search_structure:
            total_loss += self.k_structure*architecture_loss
        return total_loss,  loss, arcitecture_reduction
 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
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

    parser.add_argument('-batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='segmin')
    parser.add_argument('-model_src', type=str,  default='segment_nas_512x442_20211115_00')
    parser.add_argument('-model_dest', type=str, default='segment_nas_512x442_20211116_00')
    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-height', type=int, default=480, help='Batch image height')
    parser.add_argument('-width', type=int, default=512, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')
    parser.add_argument('-learning_rate', type=float, default=1.0e-4, help='Adam learning rate')
    parser.add_argument('-max_search_depth', type=int, default=5, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-min_search_depth', type=int, default=5, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-max_cell_steps', type=int, default=3, help='maximum number of convolution cells in layer to search/minimize')
    parser.add_argument('-channel_multiple', type=float, default=1.5, help='maximum number of layers to grow per level')
    parser.add_argument('-k_structure', type=float, default=1.0e-4, help='Structure minimization weighting fator')
    parser.add_argument('-target_structure', type=float, default=0.01, help='Structure minimization weighting fator')
    parser.add_argument('-batch_norm', type=bool, default=False)

    parser.add_argument('-prune', type=bool, default=False)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-infer', type=bool, default=True)
    parser.add_argument('-search_structure', type=bool, default=False)
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
            search_structure=args.search_structure)

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
    outputs = model(inputs)

    loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, model)

    return outputs, loss, cross_entropy_loss, architecture_loss

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    print('Network2D Test')

    import os
    import torch.optim as optim

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

        valset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
            image_paths=args.val_image_path,
            class_dictionary=class_dictionary, 
            height=args.height, 
            width=args.width, 
            imflags=args.imflags, 
            astype='float32',
            enable_transform=False)
        test_batches=int(valset.__len__()/args.batch_size)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
        test_freq = int(math.ceil(train_batches/test_batches))

    if(args.model_src and args.model_src != ''):
        modeldict = s3.GetDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modeldict is not None:
            segment = Network2d(definition=modeldict)
        else:
            print('Unable to load model definition {}/{}/{}/{}. Creating default model.'.format(
                s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
            segment = MakeNetwork2d(class_dictionary['classes'], args)

        if modelObj is not None:
            segment.load_state_dict(torch.load(io.BytesIO(modelObj)))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
            return
    else:
        # Create Default segmenter
        segment = MakeNetwork2d(class_dictionary['classes'], args)

    #I think that setting device here eliminates the need to sepcificy device in Network2D
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
    criterion = TotalLoss(k_structure=args.k_structure, target_structure=target_structure, class_weight=class_weight, search_structure=args.search_structure)
    #optimizer = optim.SGD(segment.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(segment.parameters(), lr= args.learning_rate)

    # Train
    if args.train:
        for epoch in tqdm(range(args.epochs), desc="Train epochs"):  # loop over the dataset multiple times
            iVal = iter(valloader)

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

                    data = next(iVal)
                    inputs, labels, mean, stdev = data
                    outputs, loss, cross_entropy_loss, architecture_loss = InferLoss(inputs, labels, args, segment, criterion, optimizer)

                    writer.add_scalar('loss/test', loss, int((i+1)/test_freq-1))

                    running_loss /=test_freq

                    tqdm.write('Train [{}, {:06}] cross_entropy_loss: {:0.5e}'.format(epoch + 1, i + 1, running_loss))
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

            images = np.squeeze(images.cpu().permute(0, 2, 3, 1).numpy())
            labels = np.around(np.squeeze(labels.cpu().numpy())).astype('uint8')
            segmentations = np.squeeze(segmentations.cpu().numpy()).astype('uint8')
            mean = np.squeeze(mean.numpy())
            stdev = np.squeeze(stdev.numpy())

            dsResults.infer_results(i, images, labels, segmentations, mean, stdev, dt)

            if args.fast and i+1 >= test_freq:
                break

        test_summary['objects'] = dsResults.objTypes
        test_summary['object store'] =s3def
        test_summary['results'] = dsResults.Results()
        test_summary['config'] = args.__dict__

        # If there is a way to lock this object between read and write, it would prevent the possability of loosing data
        test_path = '{}/{}/{}'.format(s3def['sets']['test']['prefix'], args.model_type, args.test_results)
        training_data = s3.GetDict(s3def['sets']['test']['bucket'], test_path)
        if training_data is None or type(training_data) is not list:
            training_data = []
        training_data.append(test_summary)
        s3.PutDict(s3def['sets']['test']['bucket'], test_path, training_data)

        test_url = s3.GetUrl(s3def['sets']['test']['bucket'], test_path)
        print("Test results {}".format(test_url))

    print('Finished network2d Test')


if __name__ == '__main__':
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        ''' https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        Launch application from console with -debug flag
        $ python3 train.py -debug
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
        Connet to vscode "Python: Remote" configuration
        '''

        debugpy.listen(address=(args.debug_address, args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    Test(args)

