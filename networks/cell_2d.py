import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from collections import OrderedDict
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
                 padding_mode='zeros'):
        super(ConvBR, self).__init__()
        self.batch_norm = batch_norm
        self.relu = relu
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
            elif self.batchnorm2d and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batchnorm2d(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

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
                 steps,
                 out_channels, 
                 in1_channels, 
                 in2_channels = 0,
                 batch_norm=True, 
                 relu=True,
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1,
                 bias=True, 
                 padding_mode='zeros'):
                
        super(Cell, self).__init__()
        self.steps = steps
        self.relu = relu

        od = OrderedDict()

        # First convolution uses in1_channels+in2_channels is input chanels. 
        # Remaining convoutions uses out_channels as chanels
        self.channel_in = in1_channels+in2_channels
        self.channel_out = out_channels

        self.conv_set_channels = ConvBR(self.channel_in, self.channel_out, batch_norm, relu, kernel_size, stride, dilation, groups, bias, padding_mode)

        for i in range(self.steps):
            conv = ConvBR(self.channel_out, self.channel_out, batch_norm, relu, kernel_size, stride, dilation, groups, bias, padding_mode)
            od['ConvBR{:02d}'.format(i)] = conv

        # 1x1 convolution to out_channels
        conv1x1 = nn.Conv2d(self.channel_out, self.channel_out, kernel_size=1)
        od['Conv1x1'.format(i)] = conv1x1

        self.cnn = torch.nn.Sequential(od)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in1, in2 = None):
        if in2 is not None:
            x = torch.cat((in1, in2))
        else:
            x = in1

        x = self.conv_set_channels(x)
        residual = x
        x = self.cnn(x)
        x = F.relu(x, inplace=True)
        x = x+residual

        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('-batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-epochs', type=int, default=4, help='Training epochs')

    args = parser.parse_args()
    return args

class Classify(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell1 = Cell(2, 8, 3)
        self.cell2 = Cell(2, 24, 8)
        self.cell3 = Cell(2, 72, 24)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(72 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.cell1(x))
        x = self.pool(self.cell2(x))
        x = self.pool(self.cell3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Test(args):
    print('Cell Test')

    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create classifier
    classify = Classify()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classify.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(classify.parameters(), lr=0.0001)

    # Train
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classify(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

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

