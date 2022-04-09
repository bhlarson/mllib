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
import torch.optim as optim
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
import cv2
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath(''))
from networks.cell2d import Cell, NormGausBasis
from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDictJson
from pymlutil.s3 import s3store, Connect
from datasets.cocostore import CocoDataset
from pymlutil.metrics import DatasetResults
# from segment.display import DrawFeatures
from torch.utils.tensorboard import SummaryWriter

# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
# https://github.com/pytorch/vision/tree/7b1b68d7142aa2070a5592d7c1a3bff3485b5ec1/torchvision/prototype/models


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
     parser.add_argument('-val_image_path', type=str, default='data/coco/val2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/deeplabv3/coco.json', help='Model class definition file.')

    parser.add_argument('-batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('-epochs', type=int, default=4, help='Training epochs')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='deeplabv3')
    parser.add_argument('-model_src', type=str,  default='segment_deeplabv3_512x442_20211118_00')
    parser.add_argument('-model_dest', type=str, default='segment_deeplabv3_512x442_20211118_01')
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

    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-infer', type=bool, default=True)
    parser.add_argument('-onnx', type=bool, default=False)

    parser.add_argument('-test_results', type=str, default='test_results.json')
    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-imStatistics', type=bool, default=False, help='Record individual image statistics')

    parser.add_argument('-tensorboard_dir', type=str, default='/store/test/segtb', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir /store/test/segtb --bind_all')
    parser.add_argument('-class_weight', type=json.loads, default=None, help='Loss class weight ') 
    
    parser.add_argument('-description', type=json.loads, default='{"description":"Deeplab V3 image segmentaiton"}', help='Test description')


    args = parser.parse_args()
    return args

def save(model, s3, s3def, args):
    out_buffer = io.BytesIO()
    model.zero_grad(set_to_none=True)
    torch.save(model.state_dict(), out_buffer)
    s3.PutObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_dest ), out_buffer)

def onnx(model, s3, s3def, args):
    import torch.onnx as torch_onnx

    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, device='cuda')
    input_names = ["image"]
    output_names = ["segmentation"]
    outputdir = '{}/{}'.format(args.test_dir,args.model_class)

    output_filename = '{}/{}.onnx'.format(outputdir, args.model_dest)
    dynamic_axes = {input_names[0] : {0 : 'batch_size'},    # variable length axes
                            output_names[0] : {0 : 'batch_size'}}

    os.makedirs(outputdir, exist_ok=True)
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

def InferLoss(inputs, labels, args, model, loss, optimizer):
    if args.cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # zero the parameter gradients
    model.zero_grad(set_to_none=True)

    outputs = model(inputs)
    pixel_loss = outputs['aux']
    outputs = outputs['out']

    loss  = loss(outputs, labels.long())

    return outputs, loss

# Classifier based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def main(args):
    print('Start train_torch')

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

    segment = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    # Resize output layer

    # Load saved model
    if(args.model_src and args.model_src != ''):
        modelObj = s3.GetObject(s3def['sets']['model']['bucket'], '{}/{}/{}.pt'.format(s3def['sets']['model']['prefix'],args.model_class,args.model_src ))

        if modelObj is not None:
            segment.load_state_dict(torch.load(io.BytesIO(modelObj)))
        else:
            print('Failed to load model_src {}/{}/{}/{}.pt  Exiting'.format(s3def['sets']['model']['bucket'],s3def['sets']['model']['prefix'],args.model_class,args.model_src))
            return

    #Set comput hardware
    segment.to(device)

    total_parameters = count_parameters(segment)

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

    cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weight)
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

                outputs, loss = InferLoss(inputs, labels, args, segment, cross_entropy_loss, optimizer)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                writer.add_scalar('loss/train', loss, i)

                if i % test_freq == test_freq-1:    # Save image and run test

                    data = next(iVal)
                    inputs, labels, mean, stdev = data
                    outputs, loss = InferLoss(inputs, labels, args, segment, cross_entropy_loss, optimizer)

                    writer.add_scalar('loss/test', loss, int((i+1)/test_freq-1))

                    running_loss /=test_freq

                    tqdm.write('Train [{}, {:06}] loss: {:0.5e}, test loss {:0.5e} '.format(epoch + 1, i + 1, running_loss, loss))
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

                        iman = cv2.putText(iman, 'Annotation',(10,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
                        imseg = cv2.putText(imseg, 'Segmentation',(10,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
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
            class_dictionary=class_dictionary, 
            image_paths=args.val_image_path,
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

        for i, data in tqdm(enumerate(testloader), total=int(testset.__len__()/args.batch_size), desc="Inference steps"): # try https://pypi.org/project/enlighten/ rather than tqdm to include progress & log messages
            images, labels, mean, stdev = data
            if args.cuda:
                images = images.cuda()

            initial = datetime.now()

            outputs = segment(images)
            pixel_loss = outputs['aux']
            outputs = outputs['out']
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
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy
        ''' https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        To debuby, launch application from console with -debug flag
        $ python3 train_torch.py -debug
        In vscode launch.json
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

    result = main(args)
    sys.exit(result)

