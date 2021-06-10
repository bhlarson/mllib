import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
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
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros'):
        super(ConvBR, self).__init__()
        self.batch_norm = batch_norm
        self.relu = relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if self.batch_norm:
            self.batchnorm2d = nn.BatachNorm2d(out_channels)

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
            x = self.batchnorm2d()
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
                 in1_channels, 
                 in2_channels,
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

        self.cnn = torch.nn.Sequential()

        for i in range(self.steps):
            conv = ConvBR(in1_channels+in2_channels, in1_channels+in2_channels, batch_norm, relu, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
            self.cnn.add_module('ConvBR{:02d}'.format(i),conv)

        # 1x1 convolution to out_channels
        conv1x1 = nn.Conv2d(in1_channels+in2_channels, in1_channels, kernel_size=1)
        self.cnn.add_module('ConvBR{:02d}'.format(i),conv1x1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in1, in2):
        if in2 is not None:
            x = torch.cat((in1, in2))
        else:
            x = in1

        x = self.cnn(x)

        if self.relu:
            x = F.relu(x, inplace=True)

        x = sum(x, in1)

        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-s3_name', type=str, default='store', help='Credential file s3 name.')
    parser.add_argument('-dataset', type=str, default='cityscapes', help='Dataset name.')
    parser.add_argument('-set', type=str, default='training', help='Set to extract from dataset')

    parser.add_argument('-classes', type=json.loads, default=None, help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('-classes_file', type=str, default='datasets/cityscapes.json', help='Class dictionary JSON file')

    args = parser.parse_args()
    return args

def Test(args):
    print('CityDataset Test')

    creds = ReadDictJson(args.credentails)
    s3_creds = next(filter(lambda d: d.get('name') == args.s3_name, creds), None)
    s3 = Connect(s3_creds)
    s3_index = s3.GetDict(s3_creds['index']['bucket'],s3_creds['index']['prefix'] )
    dataset = s3_index['sets']['dataset']

    dataset_dfn = next(filter(lambda d: d.get('name') == args.dataset, s3_index['sets']['dataset']['datasets']), None)
    dataset_index = s3.GetDict(dataset_dfn['bucket'],dataset_dfn['prefix'] )

    #dataset['prefix'] += '/{}'.format(args.dataset.replace('/', ''))
    #dataset_index_path='{}/index.json'.format(dataset['prefix'])
    #dataset_index = s3.GetDict(s3_index['sets']['dataset']['bucket'],dataset_index_path)

    if args.set is not None:
        dataset_list = list(filter(lambda d: d.get('set') == args.set, dataset_index['dataset']))
    else:
        dataset_list = dataset_index['dataset']

    CityTorch = CityDataset(s3, dataset_list, classes=args.classes)
    print('__len__() = {}'.format(CityTorch.__len__()))
    print('__getitem__() = {}'.format(CityTorch.__getitem__(0)))

if __name__ == '__main__':
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

    if args.classes is None and args.classes_file is not None :
        if '.json' in args.classes_file:
            args.classes = json.load(open(args.classes_file))
    Test(args)

