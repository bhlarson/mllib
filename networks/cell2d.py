import math
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional

sys.path.insert(0, os.path.abspath(''))
from utils.torch_util import count_parameters, model_stats, model_weights

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
                 padding_mode='zeros'):
        super(ConvBR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.relu = relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        if type(kernel_size) == int:
            padding = kernel_size // 2 # dynamic add padding based on the kernel_size
        else:
            padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if self.batch_norm:
            self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.batch_norm and self.batchnorm2d and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batchnorm2d(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    # Remove specific network dimensions
    # remove dimension where inDimensions and outDimensions arrays are 0 for channels to be removed
    def ApplyStructure(self, in_channel_mask=None, out_channel_mask=None):

        if in_channel_mask is not None:
            if len(in_channel_mask) == self.in_channels:
                self.conv.weight.data = self.conv.weight[:, in_channel_mask!=0]
                self.in_channels = len(in_channel_mask[in_channel_mask!=0])
            else:
                raise ValueError("len(in_channel_mask)={} must be equal to self.in_channels={}".format(len(in_channel_mask), self.in_channels))

        if out_channel_mask is not None:
            if len(out_channel_mask) == self.out_channels:
                self.conv.bias.data = self.conv.bias[out_channel_mask!=0]
                self.conv.weight.data = self.conv.weight[out_channel_mask!=0]

                if self.batch_norm:
                    self.batchnorm2d.bias.data = self.batchnorm2d.bias.data[out_channel_mask!=0]
                    self.batchnorm2d.weight.data = self.batchnorm2d.weight.data[out_channel_mask!=0]
                    self.batchnorm2d.running_mean = self.batchnorm2d.running_mean[out_channel_mask!=0]
                    self.batchnorm2d.running_var = self.batchnorm2d.running_var[out_channel_mask!=0]

                self.out_channels = len(out_channel_mask[out_channel_mask!=0])
            else:
                raise ValueError("len(out_channel_mask)={} must be equal to self.out_channels={}".format(len(out_channel_mask), self.out_channels))

        return out_channel_mask

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
class Cell(nn.Module):
    def __init__(self,
                 steps=1,
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
                 depth=1,
                 definition=None):
                
        super(Cell, self).__init__()

        if definition is not None:
            steps=definition['steps']
            out_channels=definition['out_channels']
            in1_channels=definition['in1_channels']
            in2_channels = definition['in2_channels']
            batch_norm=definition['batch_norm']
            relu=definition['relu']
            kernel_size=definition['kernel_size']
            stride=definition['stride']
            padding=definition['padding']
            dilation=definition['dilation']
            groups=definition['groups']
            bias=definition['bias']
            padding_mode=definition['padding_mode']
            residual=definition['residual']
            is_cuda=definition['is_cuda']
            feature_threshold=definition['feature_threshold']
            search_structure=definition['search_structure']
            depth=definition['depth']

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

        self.cnn = torch.nn.ModuleList()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convoutions uses out_channels as chanels





        self.conv_size = ConvBR(self.in1_channels+self.in2_channels, self.out_channels, batch_norm, relu, kernel_size, stride, dilation, groups, bias, padding_mode)

        for i in range(self.steps):
            conv = ConvBR(self.out_channels, self.out_channels, batch_norm, relu, kernel_size, stride, dilation, groups, bias, padding_mode)
            self.cnn.append(conv)

        # 1x1 convolution to out_channels
        self.conv1x1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self._initialize_weights()
        self.total_trainable_weights = model_weights(self)
        self.cnn_step_weights = model_weights(self.cnn[0])
        self.dimension_weights = self.cnn_step_weights/self.out_channels


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
            'cnn_step_weights': self.cnn_step_weights,
            'dimension_weights': self.dimension_weights,
        }

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
        dimension_weights = torch.squeeze(torch.linalg.norm(self.conv1x1.weight,dim=1))
        out_channel_mask = torch.where(dimension_weights>self.feature_threshold, 1, 0).type(torch.IntTensor)
        num_out_channels = len(out_channel_mask[out_channel_mask!=0])

        print('dimension_threshold {}/{} = {}'.format(num_out_channels, self.out_channels, num_out_channels/self.out_channels))

        self.conv_size.ApplyStructure(in_channel_mask=in_channel_mask, out_channel_mask=out_channel_mask)

        for i, cnn in enumerate(self.cnn):
            cnn.ApplyStructure(in_channel_mask=out_channel_mask, out_channel_mask=out_channel_mask)

        # Apply structure to conv1x1
        self.conv1x1.bias.data = self.conv1x1.bias.data[out_channel_mask!=0]
        self.conv1x1.weight.data = self.conv1x1.weight.data[:,out_channel_mask!=0]
        self.conv1x1.weight.data = self.conv1x1.weight.data[out_channel_mask!=0]
        self.out_channels = len(out_channel_mask[out_channel_mask!=0])

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
                x = x*NormGausBasis(len(self.cnn), i, self.depth)
                y = y+x # Apply structure weight
        # Frozen structure
        else:
            for i, l in enumerate(self.cnn):
                x = self.cnn[i](x) 
            y = x

        # Apply learnable chanel mask to minimize channels during architecture search
        y = self.conv1x1(y)
       
        y = F.relu(y, inplace=True)

        if self.residual:
            y = y+residual

        return y

    def ArchitectureWeights(self):
        architecture_weights = torch.zeros(1)
        if self.is_cuda:
            architecture_weights = architecture_weights.cuda()

        layer_weight = self.dimension_weights*torch.sum(torch.erf(torch.squeeze(torch.linalg.norm(self.conv1x1.weight,dim=1))))

        for i, l in enumerate(self.cnn):
            x = self.sigmoid(self.depth-i) * layer_weight
            architecture_weights += x

        return architecture_weights, self.total_trainable_weights
    
class Classify(nn.Module):
    def __init__(self, is_cuda=False):
        super().__init__()
        self.is_cuda = is_cuda
        self.cells = torch.nn.ModuleList()
        self.cells.append(Cell(6, 16, 3, is_cuda=self.is_cuda))
        self.cells.append(Cell(6, 32, 16, is_cuda=self.is_cuda))
        self.cells.append(Cell(6, 64, 32, is_cuda=self.is_cuda))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.total_trainable_weights = model_weights(self)

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
        archatecture_weights = torch.zeros(1)
        if self.is_cuda:
            archatecture_weights = archatecture_weights.cuda()
        for in_cell in self.cells:
            cell_archatecture_weights, cell_total_trainable_weights = in_cell.ArchitectureWeights()
            archatecture_weights += cell_archatecture_weights

        return archatecture_weights, self.total_trainable_weights

class CrossEntropyRuntimeLoss(torch.nn.modules.loss._WeightedLoss):
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
                 prune=None, reduction: str = 'mean') -> None:
        super(CrossEntropyRuntimeLoss, self).__init__(weight, size_average, prune, reduction)
        self.ignore_index = ignore_index
        self.k_dims = 0.01
        self.k_depth = 3.0
        self.softsign = nn.Softsign()

    def forward(self, input: torch.Tensor, target: torch.Tensor, network) -> torch.Tensor:
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        dims = []
        depths = []
        archatecture_weights, total_trainable_weights = network.ArchitectureWeights()
        architecture_loss = archatecture_weights/total_trainable_weights
        '''
        cells = network.Cells()
        for cell in cells:
            # Softsign to prune the gradient of large dimensions while maintainign gradient of small dimensions
            # Prioritize reducing dimensions with small magnitude
            # dim_weight = torch.nn.Softsign(torch.squeeze(torch.linalg.norm(cell.conv1x1.weight,dim=1)))
            dims.append(torch.squeeze(torch.linalg.norm(cell.conv1x1.weight,dim=1)))
            depths.append(cell.depth)
        all_dims = torch.cat(dims, 0)
        dims_norm = self.softsign(torch.sum(all_dims)/all_dims.shape[0])
        cell_depths = torch.tensor(depths)
        norm_depth_loss = torch.sum(cell_depths)/cell_depths.shape[0]
        total_loss = loss + self.k_dims*dims_norm + self.k_depth*norm_depth_loss
        '''
        total_loss = loss + self.k_dims*architecture_loss
        return total_loss,  loss, architecture_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('-batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('-model', type=str, default='model')
    parser.add_argument('-cuda', type=bool, default=True)

    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-prune', type=bool, default=True)
    parser.add_argument('-infer', type=bool, default=True)

    args = parser.parse_args()
    return args

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    print('Cell Test')

    import os
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create classifier
    classify = Classify(args.cuda)

    device = "cpu"
    pin_memory = False
    if args.cuda:
        device = "cuda"
        pin_memory = True

    classify.to(device)


    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    full_filename = args.model+".pt"
    compressed_filename = args.model+"_min.pt"

    if os.path.exists(full_filename):
        classify.load_state_dict(torch.load(full_filename))

    total_parameters = count_parameters(classify)

    if args.prune:
        classify.ApplyStructure()
        reduced_parameters = count_parameters(classify)
        print('Reduced parameters {}/{} = {}'.format(reduced_parameters, total_parameters, reduced_parameters/total_parameters))

    # Define a Loss function and optimizer
    criterion = CrossEntropyRuntimeLoss()
    #optimizer = optim.SGD(classify.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(classify.parameters(), lr=0.001)

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        # Train
        if args.train:
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = classify(inputs)
                # loss, cross_entropy_loss, dims_norm, all_dims, norm_depth_loss, cell_depths  = criterion(outputs, labels, classify)
                loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, classify)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += cross_entropy_loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    running_loss /=20

                    weight_std, weight_mean, bias_std, bias_mean = model_stats(classify)
                    #print('[%d, %d] cross_entropy_loss: %.3f dims_norm: %.3f' % (epoch + 1, i + 1, cross_entropy_loss, dims_norm))
                    #print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.3f}, dims: {}'.format(epoch + 1, i + 1, running_loss, dims_norm, all_dims))
                    print('[{}, {:05d}] cross_entropy_loss: {:0.3f} architecture_loss: {:0.3f} weight [m:{:0.3f} std:{:0.5f}] bias [m:{:0.3f} std:{:0.5f}]'.format(epoch + 1, i + 1, running_loss, architecture_loss.item(), weight_std, weight_mean, bias_std, bias_mean))
                    running_loss = 0.0
                
                #print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.4f}, norm_depth_loss: {:0.3f}, cell_depths: {}'.format(epoch + 1, i + 1, cross_entropy_loss, dims_norm, norm_depth_loss, cell_depths))

        if args.test:
            running_loss = 0.0
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # forward + backward + optimize
                outputs = classify(inputs)
                # loss, cross_entropy_loss, dims_norm, all_dims, norm_depth_loss, cell_depths  = criterion(outputs, labels, classify)
                loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, classify)

                # print statistics
                running_loss += cross_entropy_loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    running_loss /=20

                    weight_std, weight_mean, bias_std, bias_mean = model_stats(classify)
                    #print('[%d, %d] cross_entropy_loss: %.3f dims_norm: %.3f' % (epoch + 1, i + 1, cross_entropy_loss, dims_norm))
                    #print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.3f}, dims: {}'.format(epoch + 1, i + 1, running_loss, dims_norm, all_dims))
                    print('Test cross_entropy_loss: {:0.3f} architecture_loss: {:0.3f} weight [m:{:0.3f} std:{:0.5f}] bias [m:{:0.3f} std:{:0.5f}]'.format(running_loss, architecture_loss.item(), weight_std, weight_mean, bias_std, bias_mean))
                    running_loss = 0.0
                
                # print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.4f}, norm_depth_loss: {:0.3f}, cell_depths: {}'.format(epoch + 1, i + 1, cross_entropy_loss, dims_norm, norm_depth_loss, cell_depths))

    if args.model:
        if args.prune:
            torch.save(classify.state_dict(), compressed_filename)
        else:
            torch.save(classify.state_dict(), full_filename)

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

        debugpy.listen(address=('0.0.0.0', args.debug_port))
        # Pause the program until a remote debugger is attached

        debugpy.wait_for_client()
        print("Debugger attached")

    Test(args)

