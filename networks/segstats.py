#!/usr/bin/python3
import math
import os
import sys
import io
import json
import yaml
import platform
import time
from datetime import datetime
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard import program
import torch.profiler
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from collections import OrderedDict
from typing import Callable, Optional
from tqdm import tqdm
import cv2
from scipy import stats

from pymlutil.torch_util import count_parameters, model_stats, model_weights
from pymlutil.jsonutil import ReadDict, WriteDict, str2bool
from pymlutil.s3 import s3store, Connect
from pymlutil.functions import Exponential
from pymlutil.metrics import DatasetResults
import pymlutil.version as pymlutil_version
from pymlutil.imutil import ImUtil, ImTransform

from torchdatasetutil.cocostore import CreateCocoLoaders
from torchdatasetutil.imstore import  CreateImageLoaders
from torchdatasetutil.cityscapesstore import CreateCityscapesLoaders, CityscapesDataset
import torchdatasetutil.version as  torchdatasetutil_version

sys.path.insert(0, os.path.abspath(''))
from networks.cell2d import Cell, PlotSearch, PlotGradients
from networks.totalloss import TotalLoss, FenceSitterEjectors
from networks.network2d import load, Network2d

def DefineFeature(c, iClass, minArea = 1, maxArea = float("inf"), iContour=-1, useContour = False):

    rect = cv2.boundingRect(c)

    if rect[2]*rect[3] < minArea:
        return {}

    if useContour:

        shift = np.array((rect[0],rect[1]))
        # Create a mask of contour being evaluated
        mask = np.zeros((rect[3],rect[2]), dtype=np.uint8)
        contourMask = c - shift # Shift contour to 0
        cv2.drawContours(image=mask, contours=[contourMask], contourIdx=-1, color=(255), thickness=-1) # Draw Mask
        #cv2.imwrite( "mask.png", mask )

        M = cv2.moments(mask, True)
        area = M['m00']
        center = np.array((M['m10']/M['m00'], M['m01']/M['m00']))+shift
        center = center.tolist()

    else:
        M = cv2.moments(c)
        # Calculate the area of each contour
        area = M['m00']
        if area > 0:
            center = (M['m10']/M['m00'], M['m01']/M['m00'])
            # Ignore contours that are too small or too large
            if area < 1.0 or area < minArea or area > maxArea:
                #print('filtered class {} area={}'.format(iClass, area))
                return {}

            mu20 = M['m20']/M['m00']-center[0]*center[0]
            mu02 = M['m02']/M['m00']-center[1]*center[1]
            mu11 = M['m11']/M['m00']-center[0]*center[1]

            cov = np.array([[mu20, mu11],[mu11, mu02]])

            eigenvalue, eigenvector = np.linalg.eig(cov)
            feature = {'class':iClass, 'iContour':iContour, 'center':center,  'rect': rect, 'area':area, 'eigenvalue':eigenvalue.tolist(), 'eigenvector':eigenvector.tolist(), 'moments':M, 'covariance':cov.tolist(), 'contour':c.tolist()}
        else:
            feature = None

    
    return feature

def FilterContours(seg, objectType, filters={}, useContour = False):
    
    iClass = objectType['trainId']

    [height, width] = seg.shape[-2::]

    # Area filter
    if 'area_filter_min' in objectType:
         minArea  = objectType['area_filter_min']
    else:
        minArea = 1
    
    if 'area_filter_max' in objectType:
         maxArea  = objectType['area_filter_max']
    else:
        maxArea = height*width

    # Create binary mask of filter type
    iSeg = []
    if 'alignment' in filters:
        for filter in filters['alignment']:
            if 'roi' in filter:

                roi = np.clip(filter['roi'], a_min=0, a_max=np.max(seg.shape))
                if roi[3] > seg.shape[0]:
                    roi[0] = seg.shape[0]
                if roi[2] > seg.shape[1]:
                    roi[2] = seg.shape[1]


                if len(iSeg)==0:
                    iSeg = np.zeros_like(seg)
                    iSeg[roi[1]:roi[3],roi[0]:roi[2]] = seg[roi[1]:roi[3],roi[0]:roi[2]]
                else:
                    iSeg[roi[1]:roi[3],roi[0]:roi[2]] = seg[roi[1]:roi[3],roi[0]:roi[2]]                    
                iSeg[iSeg != iClass] = 0 # Convert to binary mask

    if len(iSeg) == 0:
        iSeg = deepcopy(seg)
        iSeg[iSeg != iClass] = 0 # Convert to binary mask        
    
    
    segFeatures = []
    contours, hierarchy = cv2.findContours(iSeg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):

        feature = DefineFeature(c, iClass, minArea, maxArea, i, useContour = useContour)
        if feature:
            segFeatures.append(feature)

    return segFeatures

def ExtractFeatures(seg, objects, filters={}, useContour = False):
    #print("values:{}".format(np.unique(seg)))
    segFeatures = {}
    obect_types = {}
    classArea = {}
    for an_object in objects:
        if an_object['trainId'] not in obect_types:
            obect_types[an_object['trainId']] = an_object
            classArea[an_object['trainId']] = 0

    for i in obect_types:
        feature_contours = FilterContours(seg, obect_types[i], filters, useContour = useContour)
        segFeatures[i] = feature_contours
        
        for feature in feature_contours:
            classArea[i] += feature['area']

    return classArea, segFeatures

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug', type=str2bool, default=False, help='Wait for debuggee attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')
    parser.add_argument('-min', action='store_true', help='Minimum run with a few iterations to test execution')
    parser.add_argument('-minimum', type=str2bool, default=False, help='Minimum run with a few iterations to test execution')
    parser.add_argument('-p' '--parameters', type=str, default=None, help='Load Parameters from file')

    parser.add_argument('-credentails', type=str, default='creds.yaml', help='Credentials file.')

    parser.add_argument('-imStatistics', type=str2bool, default=False, help='Record individual image statistics')

    parser.add_argument('-dataset', type=str, default='cityscapes', choices=['coco', 'lit', 'cityscapes'], help='Dataset')
    parser.add_argument('-dataset_path', type=str, default='/data', help='Local dataset path')

    parser.add_argument('-lit_dataset', type=str, default='data/lit/dataset.yaml', help='Image dataset file')
    parser.add_argument('-lit_class_dict', type=str, default='model/crisplit/lit.json', help='Model class definition file.')

    parser.add_argument('-coco_class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')

    parser.add_argument('-cityscapes_data', type=str, default='data/cityscapes', help='Image dataset file')
    parser.add_argument('-cityscapes_class_dict', type=str, default='model/cityscapes/cityscapes8.json', help='Model class definition file.')
    parser.add_argument('-height', type=int, default=512, help='Batch image height')
    parser.add_argument('-width', type=int, default=768, help='Batch image width')

    parser.add_argument('-batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('-num_workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('-model_type', type=str,  default='segmentation')
    parser.add_argument('-model_class', type=str,  default='cityscapes')
    parser.add_argument('-model_src', type=str,  default='crispcityscapes_20221010_102128_hiocnn0_search_structure_03')
    parser.add_argument('-unet_depth', type=int, default=5, help='number of encoder/decoder levels to search/minimize')
    parser.add_argument('-max_cell_steps', type=int, default=3, help='maximum number of convolution cells in layer to search/minimize')
    parser.add_argument('-channel_multiple', type=float, default=2, help='maximum number of layers to grow per level')
    parser.add_argument('-cuda', type=str2bool, default=True)
    parser.add_argument('-batch_norm', type=str2bool, default=False)
    parser.add_argument('-dropout', type=str2bool, default=False, help='Enable dropout')
    parser.add_argument('-dropout_rate', type=float, default=0.0, help='Dropout probability gain')
    parser.add_argument('-weight_gain', type=float, default=5.0, help='Channel convolution norm tanh weight gain')
    parser.add_argument('-sigmoid_scale', type=float, default=5.0, help='Sigmoid scale domain for convolution channels weights')
    parser.add_argument('-feature_threshold', type=float, default=0.0, help='cell tanh pruning threshold')
    parser.add_argument('-convMaskThreshold', type=float, default=0.5, help='convolution channel sigmoid level to prune convolution channels')
    parser.add_argument('-residual', type=str2bool, default=False, help='Residual convolution functions')
    parser.add_argument('-k_prune_sigma', type=float, default=1.0, help='prune basis exponential weighting factor')
    parser.add_argument('-search_flops', type=str2bool, default=True)

    parser.add_argument('-tensorboard_dir', type=str, default='/tb_logs', 
        help='to launch the tensorboard server, in the console, enter: tensorboard --logdir ./tb --bind_all')
    parser.add_argument('-tb_dest', type=str, default='crisplit_20220909_061234_hiocnn_tb_01')

    parser.add_argument('-job', action='store_true',help='Run as job')

    parser.add_argument('-resultspath', type=str, default='results.yaml')
    parser.add_argument('-prevresultspath', type=str, default=None)
    parser.add_argument('-test_dir', type=str, default=None)
    parser.add_argument('-description', type=json.loads, default='{"description":"CRISP segmentation"}', help='Test description')

    args = parser.parse_args()

    if args.d:
        args.debug = args.d
    if args.min:
        args.minimum = args.min

    return args

def Test(args, s3, s3def, class_dictionary, model, loaders, device, results):
    torch.cuda.empty_cache()
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    test_summary = {'date':date_time}

    testloader = next(filter(lambda d: d.get('set') == 'test' or d.get('set') == 'val', loaders), None)
    if testloader is None:
        raise ValueError('{} {} failed to load testloader {}'.format(__file__, __name__, args.dataset)) 

    if args.test_dir is not None:
        outputdir = '{}/{}'.format(args.test_dir,args.model_class)
        os.makedirs(outputdir, exist_ok=True)
    else:
        outputdir = None

    dsResults = DatasetResults(class_dictionary, args.batch_size, imStatistics=args.imStatistics, imgSave=outputdir)
    dtSum = 0.0
    inferTime = []
    for i, data in tqdm(enumerate(testloader['dataloader']), 
                        total=testloader['batches'], 
                        desc="Test steps", 
                        disable=args.job, 
                        bar_format='{desc:<8.5}{percentage:3.0f}%|{bar:50}{r_bar}'):
        images, labels, mean, stdev = data
        if args.cuda:
            images = images.cuda()

        initial = datetime.now()
        with torch.no_grad():
            outputs = model(images)
            segmentations = torch.argmax(outputs, 1)
        dt = (datetime.now()-initial).total_seconds()
        dtSum += dt
        inferTime.append(dt/args.batch_size)
        tqdm.write('inferTime = {}'.format(inferTime[-1]))

        images = images.cpu().permute(0, 2, 3, 1).numpy()
        labels = np.around(labels.cpu().numpy()).astype('uint8')
        segmentations = segmentations.cpu().numpy().astype('uint8')

        for iLabel in range(labels.shape[0]):
            classArea, segFeatures = ExtractFeatures(labels[iLabel], class_dictionary['objects'])
            tqdm.write('class areas = {}'.format(classArea))
            

        dsResults.infer_results(i, images, labels, segmentations, mean.numpy(), stdev.numpy(), dt)

        if args.minimum and i+1 >= 10:
            break

        test_results = dsResults.Results()

    test_summary['objects'] = dsResults.objTypes
    test_summary['object store'] =s3def
    test_summary['results'] = test_results
    test_summary['config'] = args.__dict__
    if args.ejector is not None and type(args.ejector) != str:
        test_summary['config']['ejector'] = args.ejector.value
    test_summary['system'] = results['system']
    test_summary['training_results'] = results

    if(args.tensorboard_dir is not None and len(args.tensorboard_dir) > 0 and args.tb_dest is not None and len(args.tb_dest) > 0):
        writer_path = '{}/{}/testresults.yaml'.format(args.tensorboard_dir, args.model_dest)
        WriteDict(test_summary, writer_path)

    results['test'] = test_summary['results']
    return results

def main(args): 
    print('network2d test')

    results={}
    results['config'] = args.__dict__
    results['system'] = {
        'platform':str(platform.platform()),
        'python':str(platform.python_version()),
        'numpy': str(np.__version__),
        'torch': str(torch.__version__),
        'OpenCV': str(cv2.__version__),
        'pymlutil': str(pymlutil_version.__version__),
        'torchdatasetutil':str(torchdatasetutil_version.__version__),
    }
    print('network2d system={}'.format(yaml.dump(results['system'], default_flow_style=False) ))
    print('network2d config={}'.format(yaml.dump(results['config'], default_flow_style=False) ))

    #torch.autograd.set_detect_anomaly(True)

    s3, _, s3def = Connect(args.credentails)

    results['store'] = s3def

    device = torch.device("cpu")
    if args.cuda:
        device = torch.device("cuda")

    # Load dataset
    class_dictionary = None
    dataset_bucket = s3def['sets']['dataset']['bucket']
    if args.dataset=='coco':
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.coco_class_dict)
        loaders = CreateCocoLoaders(s3, dataset_bucket, 
            class_dict=args.coco_class_dict, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cuda = args.cuda,
            height = args.height,
            width = args.width,
        )
    elif args.dataset=='lit':
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.lit_class_dict)
        loaders = CreateImageLoaders(s3, dataset_bucket, 
            dataset_dfn=args.lit_dataset,
            class_dict=args.lit_class_dict, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cuda = args.cuda,
            height = args.height,
            width = args.width,
        )
    elif args.dataset=='cityscapes':
        class_dictionary = s3.GetDict(s3def['sets']['dataset']['bucket'],args.cityscapes_class_dict)
        loaders = CreateCityscapesLoaders(s3, s3def, 
            src = args.cityscapes_data,
            dest = args.dataset_path+'/cityscapes',
            class_dictionary = class_dictionary,
            batch_size = args.batch_size, 
            num_workers=args.num_workers,
            height=args.height,
            width=args.width, 
        )

    else:
        raise ValueError("Unupported dataset {}".format(args.dataset))

    # Load number of previous batches to continue tensorboard from previous training
    prevresultspath = None
    print('prevresultspath={}'.format(args.prevresultspath))
    if args.prevresultspath and len(args.prevresultspath) > 0:
        prevresults = ReadDict(args.prevresultspath)
        if prevresults is not None:
            if 'batches' in prevresults:
                print('found prevresultspath={}'.format(yaml.dump(prevresults, default_flow_style=False)))
                results['batches'] = prevresults['batches']
            if 'initial_parameters' in prevresults:
                results['initial_parameters'] = prevresults['initial_parameters']
                results['initial_flops'] = prevresults['initial_flops']

    segment, results = load(s3, s3def, args, class_dictionary, loaders, results)

    if 'batches' not in results:
        results['batches'] = 0

    results = Test(args, s3, s3def, class_dictionary, segment, loaders, device, results)


    if args.resultspath is not None and len(args.resultspath) > 0:
        WriteDict(results, args.resultspath)


    print('Finished {}'.format(args.model_dest, yaml.dump(results, default_flow_style=False) ))
    print(json.dumps(results, indent=2))
    return 0

if __name__ == '__main__':
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach on {}:{}".format(args.debug_address, args.debug_port))
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

        Connect to vscode "Python: Remote" configuration
        '''

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client() # Pause the program until a remote debugger is attached
        print("Debugger attached")

    result = main(args)
    sys.exit(result)
