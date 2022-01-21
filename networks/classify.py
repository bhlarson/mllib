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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')

    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-learning_rate', type=float, default=0.0001, help='Training learning rate')
    parser.add_argument('-batch_size', type=int, default=50, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='Classification')
    parser.add_argument('-model_class', type=str,  default='CIFAR10')
    parser.add_argument('-model_src', type=str,  default='wrn_20220117_sgd_03')
    parser.add_argument('-model_dest', type=str, default='wrn_20220117_sgd_04')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-k_structure', type=float, default=1.0e2, help='Structure minimization weighting fator')
    parser.add_argument('-target_structure', type=float, default=1.0e-1, help='Structure minimization weighting fator')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout_rate', type=float, default=0.3, help='Dropout probabability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convoluiton channels weights')

    parser.add_argument('-prune', type=bool, default=True)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-infer', type=bool, default=True)
    parser.add_argument('-search_structure', type=bool, default=True)
    parser.add_argument('-onnx', type=bool, default=True)

    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='/store/test/nassegtb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir /store/test/nassegtb --bind_all')

    parser.add_argument('-description', type=json.loads, default='{"description":"Cell 2D NAS classification"}', help='Run description')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152], default=101, help='Run description')

    args = parser.parse_args()
    return args

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model, out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)
    #s3.PutDict(s3def['sets']['model']['bucket'], '{}/{}/{}.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), model.definition())


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

    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
            
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        #classify = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        classify = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    #sepcificy device for model
    classify.to(device)
    
    total_parameters = count_parameters(classify)


    # Define a Loss function and optimizer
    optimizer = optim.SGD(classify.parameters(), lr=args.learning_rate, momentum=0.9)
    #optimizer = optim.Adam(classify.parameters(), lr=args.learning_rate)
    loss_metric = nn.CrossEntropyLoss()
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
                outputs = classify(inputs)
                loss = loss_metric(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                writer.add_scalar('loss/train', loss, iSample)

                if i % test_freq == test_freq-1:    # Save image and run test

                    classifications = torch.argmax(outputs, 1)
                    results = (classifications == labels).float()
                    training_accuracy = torch.sum(results)/len(results)

                    data = next(iTest)
                    inputs, labels = data

                    if args.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    #optimizer.zero_grad()
                    outputs = classify(inputs)
                    classifications = torch.argmax(outputs, 1)
                    results = (classifications == labels).float()
                    test_accuracy = torch.sum(results)/len(results)

                    loss = loss_metric(outputs, labels)

                    writer.add_scalar('accuracy/train', training_accuracy, iSample)
                    writer.add_scalar('accuracy/test', test_accuracy, iSample)
                    writer.add_scalar('loss/test', loss, iSample)

                    running_loss /=test_freq
                    tqdm.write('Test [{}, {:06f}] training accuracy={:03f} test accuracy={:03f} training loss={:0.5e}, test loss={:0.5e}'.format(
                        epoch + 1, i + 1, training_accuracy, test_accuracy, running_loss, loss))
                    running_loss = 0.0

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

