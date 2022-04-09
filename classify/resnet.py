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
import cv2
from mlflow import log_metric, log_param, log_artifacts

sys.path.insert(0, os.path.abspath(''))
from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDictJson, WriteDictJson, str2bool
from pymlutil.s3 import s3store, Connect
from torch.utils.tensorboard import SummaryWriter
from networks.totalloss import TotalLoss
from networks.cell2d import Resnet, ResnetCells, Classify, PlotSearch, PlotGradients
from networks.resnet_torchvision import resnet50


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-fast', action='store_true', help='Fast run with a few iterations')
    parser.add_argument('-description', type=str, default="", help='Training description to record with resuts')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')

    parser.add_argument('-prune', type=str2bool, default=True)
    parser.add_argument('-train', type=str2bool, default=True)
    parser.add_argument('-infer', type=str2bool, default=True)
    parser.add_argument('-search_structure', type=str2bool, default=True)
    parser.add_argument('-onnx', type=str2bool, default=True)
    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-resnet_len', type=int, choices=[18, 34, 50, 101, 152], default=50, help='Run description')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-model', type=str, default='model')

    parser.add_argument('-learning_rate', type=float, default=1e-1, help='Training learning rate')
    parser.add_argument('-batch_size', type=int, default=2000, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=120, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='Classification')
    parser.add_argument('-model_class', type=str,  default='CIFAR10')
    parser.add_argument('-model_src', type=str,  default=None)
    parser.add_argument('-model_dest', type=str, default="crisp20220125_00")
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-k_structure', type=float, default=1.0e0, help='Structure minimization weighting fator')
    parser.add_argument('-target_structure', type=float, default=0.0e0, help='Structure minimization weighting fator')
    parser.add_argument('-batch_norm', type=bool, default=True)
    parser.add_argument('-dropout_rate', type=float, default=0.2, help='Dropout probabability gain')
    parser.add_argument('-weight_gain', type=float, default=11.0, help='Convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convoluiton channels weights')

    parser.add_argument('-augment_rotation', type=float, default=10.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_translate', type=float, default=10.0, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_min', type=float, default=0.8, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_scale_max', type=float, default=1.2, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_translate_x', type=float, default=0.1, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_translate_y', type=float, default=0.1, help='Input augmentation rotation degrees')
    parser.add_argument('-augment_noise', type=float, default=0.1, help='Input augmentation rotation degrees')

    parser.add_argument('-resultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-tensorboard_dir', type=str, default='./tb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir /store/test/nassegtb --bind_all')



    args = parser.parse_args()
    return args

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model, out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def main(args):
    import torchvision
    import torchvision.transforms as transforms

    test_results={}
    test_results['system'] = {
        'platform':platform.platform(),
        'python':platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'OpenCV': cv2.__version__,
    }
    print('Cell Test system={}'.format(test_results['system'] ))

    test_results['args'] = {}
    for arg in vars(args):
        log_param(arg, getattr(args, arg))
        test_results['args'][arg]=getattr(args, arg)

    print('arguments={}'.format(test_results['args']))

    torch.autograd.set_detect_anomaly(True)

    s3, creds, s3def = Connect(args.credentails)

    test_results['classes'] = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0):
        os.makedirs(args.tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)

    device = "cpu"
    pin_memory = False
    if args.cuda:
        device = torch.device('cuda')
        pin_memory = True

    # Load dataset
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=args.augment_rotation, 
            translate=(args.augment_translate_x, args.augment_translate_y), 
            scale=(args.augment_scale_min, args.augment_scale_max), 
            interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        AddGaussianNoise(0., args.augment_noise)
    ])

    trainingset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
    train_batches=int(trainingset.__len__()/args.batch_size)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
    test_batches=int(testset.__len__()/args.batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_freq = int(math.ceil(train_batches/test_batches))


    classify = resnet50(pretrained=True)

    #sepcificy device for model
    classify.to(device)



    # Define a Loss function and optimizer
    target_structure = torch.as_tensor([args.target_structure], dtype=torch.float32)
    #compute_loss = TotalLoss(args.cuda, k_structure=args.k_structure, target_structure=target_structure, search_structure=args.search_structure)
    compute_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classify.parameters(), lr=args.learning_rate)
    #optimizer = optim.Adam(classify.parameters(), lr=args.learning_rate)

    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40, 60, 80, 100], gamma=0.1)

    iSample = 0
    lr = args.learning_rate

    # Train
    test_results['train'] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'arcitecture_reduction':[]}
    test_results['test'] = {'loss':[], 'cross_entropy_loss':[], 'architecture_loss':[], 'arcitecture_reduction':[], 'accuracy':[]}
    if args.train:
        for epoch in tqdm(range(args.epochs), desc="Train epochs", disable=args.job):  # loop over the dataset multiple times
            iTest = iter(testloader)

            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader), total=trainingset.__len__()/args.batch_size, desc="Train steps", disable=args.job):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if args.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = classify(inputs)
                #loss, cross_entropy_loss, architecture_loss, arcitecture_reduction, cell_weights  = compute_loss(outputs, labels, classify)
                loss = compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                writer.add_scalar('loss/train', loss, iSample)
                #writer.add_scalar('cross_entropy_loss/train', cross_entropy_loss, iSample)
                #writer.add_scalar('architecture_loss/train', architecture_loss, iSample)
                #writer.add_scalar('arcitecture_reduction/train', arcitecture_reduction, iSample)

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

                    #loss, cross_entropy_loss, architecture_loss, arcitecture_reduction, cell_weights  = compute_loss(outputs, labels, classify)
                    loss = compute_loss(outputs, labels)

                    running_loss /=test_freq
                    #msg = '[{:3}/{}, {:6d}/{}]  accuracy: {:03f}|{:03f} loss: {:0.5e}|{:0.5e} remaining: {:0.5e} (train|test)'.format(
                    #    epoch + 1,args.epochs, i + 1, trainingset.__len__()/args.batch_size, training_accuracy, test_accuracy, running_loss, loss, arcitecture_reduction)
                    msg = '[{:3}/{}, {:6d}/{}]  accuracy: {:03f}|{:03f} loss: {:0.5e}|{:0.5e} (train|test)'.format(
                        epoch + 1,args.epochs, i + 1, trainingset.__len__()/args.batch_size, training_accuracy, test_accuracy, running_loss, loss)
                    if args.job is True:
                        print(msg)
                    else:
                        tqdm.write(msg)
                    running_loss = 0.0

                    writer.add_scalar('accuracy/train', training_accuracy, iSample)
                    writer.add_scalar('accuracy/test', test_accuracy, iSample)
                    writer.add_scalar('loss/test', loss, iSample)
                    #writer.add_scalar('cross_entropy_loss/test', cross_entropy_loss, iSample)
                    #writer.add_scalar('architecture_loss/test', architecture_loss, iSample)
                    #writer.add_scalar('arcitecture_reduction/train', arcitecture_reduction, iSample)

                    log_metric("accuracy", test_accuracy.item())

                    #cv2.imwrite('class_weights.png', plotsearch.plot(cell_weights))
                    #cv2.imwrite('gradient_norm.png', plotgrads.plot(classify))

                iSave = 2000
                if i % iSave == iSave-1:    # print every iSave mini-batches
                    save(classify, s3, s3def, args)

                if args.fast and i+1 >= test_freq:
                    break

                iSample += 1

            # scheduler1.step()
            scheduler2.step()


            '''
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            img = plotsearch.plot(cell_weights)
            is_success, buffer = cv2.imencode(".png", img, compression_params)
            img_enc = io.BytesIO(buffer).read()
            filename = '{}/{}/{}_cw.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
            s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)

            # Plot gradients before saving which clears the gradients
            img = plotgrads.plot(classify)
            is_success, buffer = cv2.imencode(".png", img)  
            img_enc = io.BytesIO(buffer).read()
            filename = '{}/{}/{}_gn.png'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest )
            s3.PutObject(s3def['sets']['model']['bucket'], filename, img_enc)
            '''

            save(classify, s3, s3def, args)

        torch.cuda.empty_cache()
        accuracy = 0.0
        iTest = iter(testloader)
        for i, data in tqdm(enumerate(testloader), total=testset.__len__()/args.batch_size, desc="Test steps"):
            inputs, labels = data

            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            with torch.no_grad():
                outputs = classify(inputs)

            classifications = torch.argmax(outputs, 1)
            results = (classifications == labels).float()
            accuracy += torch.sum(results).item()

        accuracy /= args.batch_size*int(testset.__len__()/args.batch_size)
        log_metric("test_accuracy", accuracy)
        print("test_accuracy={}".format(accuracy))
        test_results['test_accuracy'] = accuracy
        #test_results['current_weights'], test_results['original_weights'], test_results['remnent'] = classify.Parameters()

        s3.PutDict(s3def['sets']['model']['bucket'], '{}/{}/{}_results.json'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), test_results)

    if args.resultspath is not None and len(args.resultspath) > 0:
        WriteDictJson(test_results, args.resultspath)

    print('Finished cell2d Test')
    return 0

if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        import debugpy
        print("Wait for debugger attach")
        debugpy.listen(address=(args.debug_address, args.debug_port))
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = main(args)
    sys.exit(result)

