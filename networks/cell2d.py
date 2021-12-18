import math
import os
import sys
import math
import io
import json
import platform
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm

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

        if definition is not None:
            for key in definition:
                self[key] = definition[key]


        if type(kernel_size) == int:
            padding = kernel_size // 2 # dynamic add padding based on the kernel_size
        else:
            padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if self.batch_norm:
            self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self._initialize_weights()

        self.total_trainable_weights = model_weights(self)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
            'bias': self.bias,
            'padding_mode': self.padding_mode
        }

        #dfn = deepcopy(self.__dict__)
        #dfn['depth'] = self.depth.item()

        return definition_dict

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batchnorm2d(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def ArchitectureWeights(self):
        conv_weights = torch.tanh(self.weight_gain*torch.linalg.norm(self.conv.weight, dim=(1,2,3))) 
        architecture_weights = self.total_trainable_weights * torch.sum(conv_weights) / self.out_channels

        return architecture_weights, self.total_trainable_weights

    # Remove specific network dimensions
    # remove dimension where inDimensions and outDimensions arrays are 0 for channels to be removed
    def ApplyStructure(self, in_channel_mask=None, out_channel_mask=None):
        if in_channel_mask is not None:
            if len(in_channel_mask) == self.in_channels:
                self.conv.weight.data = self.conv.weight[:, in_channel_mask!=0]
                self.in_channels = len(in_channel_mask[in_channel_mask!=0])
            else:
                raise ValueError("len(in_channel_mask)={} must be equal to self.in_channels={}".format(len(in_channel_mask), self.in_channels))

        # Convolution norm gain mask
        print("ApplyStructure convolution norm {}".format(torch.linalg.norm(self.conv.weight, dim=(1,2,3))))
        conv_mask = torch.tanh(self.weight_gain*torch.linalg.norm(self.conv.weight, dim=(1,2,3))) 
        conv_mask = conv_mask > self.convMaskThreshold

        if out_channel_mask is not None:
            if len(out_channel_mask) == self.out_channels:
                conv_mask = conv_mask[out_channel_mask!=0]
                
            else:
                raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))

        pruned_convolutions = len(conv_mask[conv_mask==False])
        if pruned_convolutions > 0:
            numconvolutions = len(conv_mask)
            print("Pruned {}={}/{} convolutional weights".format(pruned_convolutions/numconvolutions, pruned_convolutions, numconvolutions))

        self.conv.bias.data = self.conv.bias[conv_mask!=0]
        self.conv.weight.data = self.conv.weight[conv_mask!=0]

        if self.batch_norm:
            self.batchnorm2d.bias.data = self.batchnorm2d.bias.data[conv_mask!=0]
            self.batchnorm2d.weight.data = self.batchnorm2d.weight.data[conv_mask!=0]
            self.batchnorm2d.running_mean = self.batchnorm2d.running_mean[conv_mask!=0]
            self.batchnorm2d.running_var = self.batchnorm2d.running_var[conv_mask!=0]

        self.out_channels = len(conv_mask[conv_mask!=0])

        return conv_mask



# Inner neural architecture cell search with convolution steps batch norm parameters
# Process:
# 1) Concatenate inputs
# 2) Repeat ConvBR "steps" times
# 3) 1x1 convolution to return to in1 number of channels
# 4) Sum with in1 (residual bypass)
# in1 and in2: [Batch, channel, height width]
# in_channels = in1 channels + in2 channels
# out_channels = 
# steps: integer 1..n
# batchNorm: true/false

DefaultMaxDepth = 3
class Cell(nn.Module):
    def __init__(self,
                 steps=DefaultMaxDepth,
                 out_channels=1, 
                 in1_channels=3, 
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
                 depth=DefaultMaxDepth,
                 weight_gain = 11.0,
                 convMaskThreshold=0.5,
                 definition=None,
                 ):
                
        super(Cell, self).__init__()

        self.steps = steps
        self.out_channels = out_channels
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
        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.float))
        self.weight_gain = weight_gain
        self.convMaskThreshold = convMaskThreshold

        if definition is not None:
            for key in definition:
                self.__dict__[key] = definition[key]
                
            if 'depth' in definition:
                self.depth = nn.Parameter(torch.tensor(definition['depth'], dtype=torch.float))

        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convoutions uses out_channels as chanels

        convdfn = None
        if definition is not None and 'conv_size' in definition:
            convdfn = definition['conv_size']

        self.conv_size = ConvBR(self.in1_channels+self.in2_channels, self.out_channels, 
            batch_norm=batch_norm, 
            relu=relu, 
            kernel_size=kernel_size, 
            stride=stride, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode,
            weight_gain=self.weight_gain,
            convMaskThreshold=convMaskThreshold, 
            definition=convdfn)

        for i in range(self.steps):
            convdfn = None
            if definition is not None and 'cnn' in definition and i < len(definition['cnn']):
                convdfn = definition['cnn'][i]

            conv = ConvBR(self.out_channels, self.out_channels, 
                batch_norm=batch_norm, 
                relu=relu, 
                kernel_size=kernel_size, 
                stride=stride, 
                dilation=dilation, 
                groups=groups, 
                bias=bias, 
                padding_mode=padding_mode,
                weight_gain=weight_gain,
                convMaskThreshold=convMaskThreshold, 
                definition=convdfn)
            self.cnn.append(conv)

        convdfn = None
        if definition is not None and 'conv1x1' in definition:
            convdfn = definition['conv1x1']

        # 1x1 convolution to out_channels
        self.conv1x1 = ConvBR(self.out_channels, self.out_channels, batch_norm=False, relu=True, kernel_size=1, definition=convdfn)

        self._initialize_weights()
        self.total_trainable_weights = model_weights(self)


    def definition(self):
        definition_dict = {
            'steps': self.steps,
            'out_channels': self.out_channels, 
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
            'depth': self.depth.item(),
            'total_trainable_weights': self.total_trainable_weights,
        }

        definition_dict['conv_size'] = self.conv_size.definition()
        definition_dict['cnn'] = []
        for conv in self.cnn:
            definition_dict['cnn'].append(conv.definition())
        definition_dict['conv1x1'] = self.conv1x1.definition()

        #dfn = deepcopy(self.__dict__)
        #dfn['depth'] = self.depth.item()

        return definition_dict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def ApplyStructure(self, in1_channel_mask=None, in2_channel_mask=None):

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
        
        # Fix convolution depth
        cnn = torch.nn.ModuleList()
        if self.depth.item() < 1:
            cnn_depth = 1
        elif round(self.depth.item()) < self.steps:
            cnn_depth = round(self.depth.item())
        else:
            cnn_depth = self.steps
        print('cell_depth {}/{} = {}'.format(cnn_depth, self.steps,cnn_depth/self.steps))
        self.cnn = self.cnn[0:cnn_depth]
        self.steps = cnn_depth

        # Drop minimized dimensions      
        out_channel_mask = self.conv_size.ApplyStructure(in_channel_mask=in_channel_mask)

        for i, cnn in enumerate(self.cnn):
            out_channel_mask = cnn.ApplyStructure(in_channel_mask=out_channel_mask)


        # Apply structure to conv1x1
        out_channel_mask = self.conv1x1.ApplyStructure(in_channel_mask=out_channel_mask)

        #print('dimension_threshold {}/{} = {}'.format(num_out_channels, self.out_channels, num_out_channels/self.out_channels))

        return out_channel_mask


    def forward(self, in1, in2 = None):
        if in2 is not None:
            x = torch.cat((in1, in2), dim=1)
        else:
            x = in1

        # Resizing convolution
        x = self.conv_size(x)
        residual = x
        # Learnable number of convolution layers
        # Continuous relaxation of number of layers through a basis function providing continuous search space
        if self.search_structure:
            y = torch.zeros_like(x)
            for i, l in enumerate(self.cnn):
                x = self.cnn[i](x)
                y += x*NormGausBasis(len(self.cnn), i, self.depth) # Weight output

        # Frozen structure
        else:
            for i, l in enumerate(self.cnn):
                x = self.cnn[i](x) 
            y = x

        # Apply learnable chanel mask to minimize channels during architecture search
        y = self.conv1x1(y)


        if self.residual:
            y = y+residual

        return y

    def ArchitectureWeights(self):
        architecture_weights = torch.tensor(0.0)
        if self.is_cuda:
            architecture_weights = architecture_weights.cuda()

        layer_weight, _ = self.conv_size.ArchitectureWeights()
        architecture_weights += layer_weight

        layer_weights = []
        cell_weights = []
        for i, l in enumerate(self.cnn): 
            layer_weight, _ = l.ArchitectureWeights()
            layer_weights.append(layer_weight)
            cell_weights.append(NormGausBasis(len(self.cnn), i, self.depth))

        max_value = max(cell_weights)
        max_index = cell_weights.index(max_value)

        for i, l in enumerate(self.cnn): 
            if i < max_index :
                architecture_weights += layer_weights[i]
            else:
                architecture_weights += cell_weights[i]*layer_weights[i]

        layer_weight, _ = self.conv1x1.ArchitectureWeights()
        architecture_weights += layer_weight

        return architecture_weights, self.total_trainable_weights

class Classify(nn.Module):
    def __init__(self, is_cuda=False, source_channels = 3, out_channels = 10, initial_channels=16, weight_gain=11, definition=None):
        super().__init__()
        self.is_cuda = is_cuda
        self.source_channels = source_channels
        self.out_channels = out_channels
        self.initial_channels = initial_channels
        self.weight_gain = weight_gain

        if definition is not None:
            for key in definition:
                self.__dict__[key] = definition[key]
                
        self.cells = torch.nn.ModuleList()

        convdfn = None
        if definition is not None and 'cells' in definition:
            if len(definition['cells']) > 0:
                convdfn = definition['cells'][0]
        self.cells.append(Cell(out_channels=self.initial_channels, in1_channels=self.source_channels, is_cuda=self.is_cuda,  weight_gain = self.weight_gain, definition=convdfn))

        convdfn = None
        if definition is not None and 'cells' in definition:
            if len(definition['cells']) > 1:
                convdfn = definition['cells'][1]       
        self.cells.append(Cell(out_channels=32, in1_channels=16, is_cuda=self.is_cuda,  weight_gain = self.weight_gain, definition=convdfn))

        convdfn = None
        if definition is not None and 'cells' in definition:
            if len(definition['cells']) > 2:
                convdfn = definition['cells'][2] 
        self.cells.append(Cell(out_channels=64, in1_channels=32, is_cuda=self.is_cuda,  weight_gain = self.weight_gain, definition=convdfn))

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, self.out_channels)

        self.total_trainable_weights = model_weights(self)

        self.fc_weights = model_weights(self.fc1)+model_weights(self.fc2)+model_weights(self.fc3)

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
            #'depth': self.depth.item(),
            'cells': [],
        }

        for ed in self.cells:
            definition_dict['cells'].append(ed.definition())

        architecture_weights, total_trainable_weights = self.ArchitectureWeights()
        definition_dict['architecture_weights']= architecture_weights.item()
        definition_dict['total_trainable_weights']= total_trainable_weights

        return definition_dict

    def ApplyStructure(self):
        in_channel_mask = None
        for cell in self.cells:
            in_channel_mask = cell.ApplyStructure(in1_channel_mask=in_channel_mask)

        # Remove pruned weights from fc1
        if in_channel_mask is not None:
            fc1weights = torch.reshape(self.fc1.weight.data,(256,64,4,4))
            fc1weights = fc1weights[:,in_channel_mask!=0]
            self.fc1.weight.data = torch.flatten(fc1weights, 1)

    def forward(self, x):
        for cell in self.cells:
            x = self.pool(cell(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def Cells(self):
        return self.cells

    def ArchitectureWeights(self):
        archatecture_weights = torch.tensor(0.0)+self.fc_weights
        if self.is_cuda:
            archatecture_weights = archatecture_weights.cuda()
        for in_cell in self.cells:
            cell_archatecture_weights, cell_total_trainable_weights = in_cell.ArchitectureWeights()
            archatecture_weights += cell_archatecture_weights

        return archatecture_weights, self.total_trainable_weights

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='Classification')
    parser.add_argument('-model_class', type=str,  default='CIFAR10')
    parser.add_argument('-model_src', type=str,  default='class_nas_bn_20211217_00')
    parser.add_argument('-model_dest', type=str, default='class_nas_bn_20211217_01')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-k_structure', type=float, default=1.0e-1, help='Structure minimization weighting fator')
    parser.add_argument('-target_structure', type=float, default=0.1, help='Structure minimization weighting fator')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-weight_gain', type=float, default=5.0, help='Convolution norm tanh weight gain')

    parser.add_argument('-prune', type=bool, default=False)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-infer', type=bool, default=True)
    parser.add_argument('-search_structure', type=bool, default=True)
    parser.add_argument('-onnx', type=bool, default=False)

    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='/store/test/nassegtb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir /store/test/nassegtb --bind_all')

    parser.add_argument('-description', type=json.loads, default='{"description":"Cell 2D NAS classification"}', help='Run description')

    args = parser.parse_args()
    return args

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model, out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)
    s3.PutDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), model.definition())


# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    import torchvision
    import torchvision.transforms as transforms

    print('Cell Test')

    system = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'numpy version': sys.modules['numpy'].__version__,
    }

    print('system={}'.format(system))

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
        modeldict = s3.GetDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        #if modeldict is not None:
        #    classify = Classify(is_cuda=args.cuda, definition=modeldict)
        #else:
        #    print('Unable to load model definition {}/{}/{}/{}. Creating default model.'.format(
        #        s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
        #    classify = Classify(is_cuda=args.cuda)

        if modelObj is not None:
            classify = torch.load(io.BytesIO(modelObj))
        else:
            print('Failed to load model {}. Exiting.'.format(args.model_src))
            return -1
    else:
        # Create Default segmenter
        classify = Classify(is_cuda=args.cuda, weight_gain=args.weight_gain)

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
    optimizer = optim.Adam(classify.parameters(), lr=0.001)
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
                outputs = classify(inputs)
                loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, classify)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                writer.add_scalar('loss/train', loss, i)
                writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, i)
                writer.add_scalar('architecture_loss/train', architecture_loss, i)

                if i % test_freq == test_freq-1:    # Save image and run test

                    data = next(iTest)
                    inputs, labels = data

                    if args.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    optimizer.zero_grad()
                    outputs = classify(inputs)
                    loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, classify)

                    writer.add_scalar('loss/test', loss, int((i+1)/test_freq-1))
                    writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, int((i+1)/test_freq-1))
                    writer.add_scalar('architecture_loss/test', architecture_loss, int((i+1)/test_freq-1))

                    running_loss /=test_freq
                    tqdm.write('Train [{}, {:06}] loss: {:0.5e} architecture_loss: {:0.5e}'.format(epoch + 1, i + 1, running_loss, architecture_loss))
                    running_loss = 0.0

                iSave = 2000
                if i % iSave == iSave-1:    # print every 20 mini-batches
                    save(classify, s3, s3def, args)

                if args.fast and i+1 >= test_freq:
                    break

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

