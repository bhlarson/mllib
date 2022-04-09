import os
import sys
import argparse
import json
import random
import math
import glob
import io
import cv2
import numpy as np
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

sys.path.insert(0, os.path.abspath(''))
from pymlutil.jsonutil import WriteDictJson, ReadDictJson
from pymlutil.s3 import s3store, Connect

def str_prune(text, prefix, suffix):
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.endswith(suffix): 
        text = text[:len(text)-len(suffix)]
    return text

class CityDataset(Dataset):

    def __init__(self, s3, dataset_index, classes=None, normalize=True, transform=True, flipX=False, flipY=True, rotate=15, scale_min=.75, scale_max=1.25, offset=0.1 ):
        print('CityDataset __init__')
        self.s3=s3
        self.dataset_index=dataset_index
        self.classes = classes

        self.normalize = normalize
        self.transform = transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset

        # Create a list of elements from dataset
        data_lists = {}

        for set in tqdm(dataset_index):
            data_list = self.s3.ListObjects(set['bucket'], setname=set['prefix'], pattern=set['pattern'], recursive=set['recursive'])
            for entry in tqdm(data_list, leave=False):
                basename = os.path.basename(entry)
                basename = str_prune(basename, set['match']['prefix'], set['match']['suffix'])
                if basename not in data_lists:
                    data_lists[basename] = {'name':basename, 'bucket':set['bucket']}
                data_lists[basename][set['type']] = entry

        self.data = list(data_lists.values())

    def __len__(self):
        return len(self.data)

    def DecodeImage(self, bucket, objectname, flags=cv2.IMREAD_COLOR):
        img = None
        imgbuff = self.s3.GetObject(bucket, objectname)
        if imgbuff:
            imgbuff = np.fromstring(imgbuff, dtype='uint8')
            img = cv2.imdecode(imgbuff, flags=flags)
        return img



    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])
        if 'image' in data:
            image = self.DecodeImage(data['bucket'], data['image'])
            if image is not None:
                data['image_buffer'] = image
        if 'label' in data:
            label = self.DecodeImage(data['bucket'], data['label'], cv2.IMREAD_GRAYSCALE)
            if label is not None:
                # convert label lable id to trainId pixel value
                if self.classes is not None:
                    for objecttype in self.classes['objects']:
                        label[label == objecttype['id']] = objecttype['trainId']

                data['label_buffer'] = label
        if 'instance' in data:
            instance = self.DecodeImage(data['bucket'], data['label'], cv2.IMREAD_GRAYSCALE)
            if instance is not None:
                data['instance_buffer'] = instance

        if self.transform:
            height, width = image.shape[:2]

            scale = np.random.uniform(self.scale_min, self.scale_max)
            angle = np.random.uniform(-self.rotate, self.rotate)
            offsetX = width*np.random.uniform(-self.offset, self.offset)
            offsetY = height*np.random.uniform(-self.offset, self.offset)
            center = (width/2.0 + offsetX, height/2.0 + offsetY)
            mat = cv2.getRotationMatrix2D(center, angle, scale)

            image = cv2.warpAffine(src=image, M=mat, dsize=(width, height))
            label = cv2.warpAffine(src=label, M=mat, dsize=(width, height))

        return data

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

    s3, creds, s3_creds = Connect(args.credentails, args.s3_name)

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

