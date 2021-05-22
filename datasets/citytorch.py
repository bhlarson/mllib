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
from utils.jsonutil import WriteDictJson, ReadDictJson

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store, Connect

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-ann_dir', type=str, default='ann')
    parser.add_argument('-record_dir', type=str, default='cityrecord', help='Path record work directory')

    parser.add_argument('-sets', type=json.loads,
        default='[{"name":"train"}, {"name":"val"}, {"name":"test"}]',
        help='Json string containing an array of [{"name":"<>", "ratio":<probability>}]')

    defaultsetname = '{}-cityscape'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('-setname', type=str, default=defaultsetname, help='Path to training set directory')
   
    parser.add_argument('-classes', type=json.loads, default='{}', help='Class dictionary JSON.  Leave empty if classes_file points to a JSON file.')
    parser.add_argument('-classes_file', type=str, default='datasets/cityscapes.json', help='Class dictionary JSON file')

    parser.add_argument('-debug', action='store_true',help='Wait for debugge attach')
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-s3_name', type=str, default='mllib-s3', help='Credential file s3 name.')
    parser.add_argument('-dataset', type=str, default='cityscapes', help='Dataset name.')
    parser.add_argument('-set', type=str, default='training', help='Set to extract from dataset')

    args = parser.parse_args()
    return args

def str_prune(text, prefix, suffix):
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.endswith(suffix): 
        text = text[:len(text)-len(suffix)]
    return text

class CityDataset(Dataset):

    def list_iman(self, imgs, anns):

        for img in imgs:
            interDir = remove_prefix(img, imgPath)
            interDir = remove_suffix(interDir, args.image_extension)

            ann = '{}{}{}'.format(annPaths[iPath],interDir,args.annotation_extension)
            if(os.path.exists(img) and os.path.exists(ann)):
                iman[dataset['name']].append({'im':img, 'an':ann})

        return iman  

    def __init__(self, s3, dataset, dataset_index, transform=None, target_transform=None):
        print('CityDataset __init__')
        self.s3=s3
        self.dataset=dataset
        self.dataset_index=dataset_index
        self.transform=transform
        self.target_transform=target_transform

        # Create a list of elements from dataset
        data_lists = {}

        for set in tqdm(dataset_index):
            setname = '{}/{}'.format(self.dataset['prefix'],set['setname'] )
            data_list = self.s3.ListObjects(self.dataset['bucket'], setname=setname, pattern=set['pattern'], recursive=set['recursive'])
            for entry in tqdm(data_list, leave=False):
                basename = os.path.basename(entry)
                basename = str_prune(basename, set['match']['prefix'], set['match']['suffix'])
                if basename not in data_lists:
                    data_lists[basename] = {'name':basename}
                data_lists[basename][set['type']] = entry

        self.data = list(data_lists.values())

    def __len__(self):
        return len(self.data)

    def DecodeImage(self, objectname, flags=cv2.IMREAD_COLOR):
        img = None
        imgbuff = self.s3.GetObject(self.dataset['bucket'], objectname)
        if imgbuff:
            imgbuff = np.fromstring(imgbuff, dtype='uint8')
            img = cv2.imdecode(imgbuff, flags=flags)
        return img



    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])
        if 'image' in data:
            image = self.DecodeImage(data['image'])
            if image is not None:
                data['image_buffer'] = image
        if 'label' in data:
            label = self.DecodeImage(data['label'], cv2.IMREAD_GRAYSCALE)
            if label is not None:
                data['label_buffer'] = label
        if 'instance' in data:
            instance = self.DecodeImage(data['label'], cv2.IMREAD_GRAYSCALE)
            if instance is not None:
                data['instance_buffer'] = instance
        return self.data[idx]

def Test(args):
    print('CityDataset Test')

    creds = ReadDictJson(args.credentails)
    s3_creds = next(filter(lambda d: d.get('name') == args.s3_name, creds['s3']), None)
    s3 = Connect(s3_creds)
    s3_index = s3.GetDict(s3_creds['index']['bucket'],s3_creds['index']['prefix'] )
    dataset = s3_index['sets']['dataset']
    dataset['prefix'] += '/{}'.format(args.dataset.replace('/', ''))
    dataset_index_path='{}/index.json'.format(dataset['prefix'])
    dataset_index = s3.GetDict(s3_index['sets']['dataset']['bucket'],dataset_index_path)
    if args.set is not None:
        dataset_list = list(filter(lambda d: d.get('set') == args.set, dataset_index['dataset']))
    else:
        dataset_list = dataset_index['dataset']

    CityTorch = CityDataset(s3, dataset, dataset_list)
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



    Test(args)

