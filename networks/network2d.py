import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
import cv2

sys.path.insert(0, os.path.abspath(''))
from networks.cell2d import Cell
from utils.torch_util import count_parameters, model_stats, model_weights
from utils.jsonutil import ReadDictJson
from utils.s3 import s3store, Connect
from datasets.cocostore import CocoDataset

class Network2d(nn.Module):
    def __init__(self, out_channels, source_channels=3, initial_channels=64, is_cuda=True, search_depth=7, max_cell_steps=6, channel_multiple=1.5, cell=Cell):
        super(Network2d, self).__init__()

        self.search_depth = search_depth
        self.out_channels = out_channels
        self.source_channels = source_channels
        self.initial_channels = initial_channels
        self.is_cuda = is_cuda
        self.cell = cell
        self.max_cell_steps = max_cell_steps
        self.depth = torch.nn.Parameter(torch.ones(1)*search_depth) # Initial depth parameter = search_depth

        self.encode_decode = torch.nn.ModuleList()
        self.upsample = torch.nn.ModuleList()
        self.final_conv = torch.nn.ModuleList()

        encoder_channels = initial_channels
        prev_encoder_chanels = source_channels
        feedforward_chanels = []

        for i in range(self.search_depth-1):
            self.encode_decode.append(cell(max_cell_steps, encoder_channels, prev_encoder_chanels, is_cuda=self.is_cuda))
            feedforward_chanels.append(encoder_channels)
            prev_encoder_chanels = encoder_channels
            encoder_channels = int(channel_multiple*encoder_channels)

        self.encode_decode.append(cell(max_cell_steps, encoder_channels, prev_encoder_chanels, is_cuda=self.is_cuda))

        for i in range(self.search_depth-1):
            self.upsample.append(nn.ConvTranspose2d(encoder_channels, encoder_channels, 2, stride=2))

            prev_encoder_chanels = encoder_channels
            encoder_channels = int(encoder_channels/channel_multiple)
            self.encode_decode.append(cell(max_cell_steps, encoder_channels, prev_encoder_chanels, feedforward_chanels[-(i+1)], is_cuda=self.is_cuda))       

        self.encode_decode.append(cell(max_cell_steps, out_channels, encoder_channels, is_cuda=self.is_cuda))
        self.pool = nn.MaxPool2d(2, 2)

    def GaussianBasis(self, i, a, r=1.0):
        return torch.exp(-1*torch.square(r*(i-a)))

    def NormGausBasis(self, i, a, x, r=1.0):
        den = torch.nn.Parameter(torch.zeros(1))
        if x.is_cuda:
            den = den.cuda()
        for j, l in enumerate(self.encode_decode):
            den = den + self.GaussianBasis(j,a,r)
        return torch.mul(torch.exp(-1*torch.square(r*(i-a)))/den, x)

    def forward(self, x):
        y = torch.zeros(self.search_depth)

        feed_forward = []
        for i in range(self.search_depth-1):
            x = self.encode_decode[i](x)
            feed_forward.append(x)
            x = self.pool(x)

        # Feed-through
        x = self.encode_decode[self.search_depth-1](x)

        for i in range(self.search_depth-1):
            x = self.upsample[i](x)
            x = self.encode_decode[i+self.search_depth](x, feed_forward[-(i+1)])


        x = self.encode_decode[-1](x) # Size to output

        # Continuous relaxation to select network depth
        # Normalized gaussian bias continuous weighting of depths
        #x = self.NormGausBasis(i,self.depth, x)
        # Sum of weightings for each depth
        #y = y+x
            
        #return y
        return x

    def ApplyStructure(self, in_channels=None):
        print('ApplyStructure')

    def ArchitectureWeights(self):
        #print('ArchitectureWeights')

        archatecture_weights = torch.zeros(1)
        total_trainable_weights = torch.tensor(model_weights(self))

        if self.is_cuda:
            archatecture_weights = archatecture_weights.cuda()
            total_trainable_weights = total_trainable_weights.cuda()

        for j in range(self.search_depth-1):
            
            encode_archatecture_weights, _ = self.encode_decode[j].ArchitectureWeights()
            decode_archatecture_weights, _ = self.encode_decode[-(j+2)].ArchitectureWeights()
            resize_archatecture_weights = torch.tensor(model_weights(self.upsample[j]))

            # Sigmoid weightingof architecture weighting to reduce model depth 
            depth_weighted_archatecture_weight = F.sigmoid(self.depth-j)*(encode_archatecture_weights+decode_archatecture_weights+resize_archatecture_weights)
            archatecture_weights += depth_weighted_archatecture_weight

        return archatecture_weights, total_trainable_weights


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-trainingset', type=str, default='data/coco/annotations/instances_train2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-validationset', type=str, default='data/coco/annotations/instances_val2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-train_image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')

    parser.add_argument('-batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('-model', type=str, default='segment_nas')
    parser.add_argument('-reduce', action='store_true', help='Compress network')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-height', type=int, default=320, help='Batch image height')
    parser.add_argument('-width', type=int, default=320, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')

    args = parser.parse_args()
    return args

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
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
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
                 reduce=None, reduction: str = 'mean') -> None:
        super(CrossEntropyRuntimeLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.k_dims = 0.01
        self.k_depth = 3.0
        self.softsign = nn.Softsign()

    def forward(self, input: torch.Tensor, target: torch.Tensor, network) -> torch.Tensor:
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        #loss = F.cross_entropy(input, target, weight=self.weight,reduction=self.reduction)
        loss = F.cross_entropy(input, torch.squeeze(target).long())

        dims = []
        depths = []
        archatecture_weights, total_trainable_weights = network.ArchitectureWeights()
        architecture_loss = archatecture_weights/total_trainable_weights

        total_loss = loss + self.k_dims*architecture_loss
        return total_loss,  loss, architecture_loss


# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    print('Cell Test')

    import os
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    creds = ReadDictJson(args.credentails)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))
        return False
    s3def = creds['s3'][0]
    s3 = Connect(s3def)

    # Load dataset
    device = "cpu"
    pin_memory = False
    if args.cuda:
        device = "cuda"
        pin_memory = True

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainingset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.trainingset, 
        image_paths=args.train_image_path,
        class_dictionary=args.class_dict, 
        height=args.height, 
        width=args.width, 
        imflags=args.imflags, 
        torch_transform=transform, 
        astype='float32')

    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    valset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], dataset_desc=args.validationset, 
        image_paths=args.val_image_path,
        class_dictionary=args.class_dict, 
        height=args.height, 
        width=args.width, 
        imflags=args.imflags, 
        torch_transform=transform, 
        astype='float32')
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    '''transform = {
        'train': transforms.Compose([
            transforms.CenterCrop(500),
            #transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(500),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }'''

    # Create classifier
    segment = Network2d(trainingset.coco.objDict['classes'], is_cuda=args.cuda)

    #I think that setting device here eliminates the need to sepcificy device in Network2D
    segment.to(device)


    full_filename = args.model+".pt"
    compressed_filename = args.model+"_min.pt"

    if os.path.exists(full_filename):
        segment.load_state_dict(torch.load(full_filename))

    total_parameters = count_parameters(segment)

    if args.reduce:
        segment.ApplyStructure()
        reduced_parameters = count_parameters(segment)
        print('Reduced parameters {}/{} = {}'.format(reduced_parameters, total_parameters, reduced_parameters/total_parameters))

    # Define a Loss function and optimizer
    criterion = CrossEntropyRuntimeLoss()
    #optimizer = optim.SGD(segment.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(segment.parameters(), lr=0.001)

    # Train
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, mean, stdev = data

            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = segment(inputs)
            # loss, cross_entropy_loss, dims_norm, all_dims, norm_depth_loss, cell_depths  = criterion(outputs, labels, segment)
            loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, segment)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += cross_entropy_loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                running_loss /=20

                weight_std, weight_mean, bias_std, bias_mean = model_stats(segment)
                #print('[%d, %d] cross_entropy_loss: %.3f dims_norm: %.3f' % (epoch + 1, i + 1, cross_entropy_loss, dims_norm))
                #print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.3f}, dims: {}'.format(epoch + 1, i + 1, running_loss, dims_norm, all_dims))
                print('[{}, {:05d}] cross_entropy_loss: {:0.3f} architecture_loss: {:0.3f} weight [m:{:0.3f} std:{:0.5f}] bias [m:{:0.3f} std:{:0.5f}]'.format(
                    epoch + 1, i + 1, running_loss, architecture_loss.item(), weight_std, weight_mean, bias_std, bias_mean))
                running_loss = 0.0
            
            #print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.4f}, norm_depth_loss: {:0.3f}, cell_depths: {}'.format(epoch + 1, i + 1, cross_entropy_loss, dims_norm, norm_depth_loss, cell_depths))

        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, mean, stdev = data
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward + backward + optimize
            outputs = segment(inputs)
            # loss, cross_entropy_loss, dims_norm, all_dims, norm_depth_loss, cell_depths  = criterion(outputs, labels, segment)
            loss, cross_entropy_loss, architecture_loss  = criterion(outputs, labels, segment)

            # print statistics
            running_loss += cross_entropy_loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                running_loss /=20

                weight_std, weight_mean, bias_std, bias_mean = model_stats(segment)
                #print('[%d, %d] cross_entropy_loss: %.3f dims_norm: %.3f' % (epoch + 1, i + 1, cross_entropy_loss, dims_norm))
                #print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.3f}, dims: {}'.format(epoch + 1, i + 1, running_loss, dims_norm, all_dims))
                print('Test cross_entropy_loss: {:0.3f} architecture_loss: {:0.3f} weight [m:{:0.3f} std:{:0.5f}] bias [m:{:0.3f} std:{:0.5f}]'.format(
                    running_loss, architecture_loss.item(), weight_std, weight_mean, bias_std, bias_mean))
                running_loss = 0.0
            
            # print('[{}, {:05d}] cross_entropy_loss: {:0.3f} dims_norm: {:0.4f}, norm_depth_loss: {:0.3f}, cell_depths: {}'.format(epoch + 1, i + 1, cross_entropy_loss, dims_norm, norm_depth_loss, cell_depths))

    if args.model:
        if args.reduce:
            torch.save(segment.state_dict(), compressed_filename)
        else:
            torch.save(segment.state_dict(), full_filename)

    print('Finished Training')


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

