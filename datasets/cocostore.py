import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store, Connect
from utils.jsonutil import ReadDictJson

class CocoStore:

    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=cv2.IMREAD_COLOR, name_deccoration=''):
        self.s3 = s3
        self.bucket = bucket
        self.dataset_desc = dataset_desc
        self.class_dictionary = class_dictionary
        self.image_paths = image_paths
        self.name_deccoration = name_deccoration
        self.imflags = imflags

        self.objDict = s3.GetDict(bucket,class_dictionary)
        self.dataset = s3.GetDict(bucket,dataset_desc)


        self.CreateIndex()
        self.i = 0

    def CreateIndex(self):
        # create index objDict rather than coco types
        anns, cats, imgs = {}, {}, {}
        imgToAnns = defaultdict(list)
        catToImgs = defaultdict(list)
        catToObj = {}
        objs = {}
        # make list based on 
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        if self.objDict is not None:
            for obj in self.objDict['objects']:
                catToObj[obj['id']] = obj
                objs[obj['trainId']] = {'category':obj['category'], 
                                        'color':obj['color'],
                                        'display':obj['display']
                                        }


        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.catToObj = catToObj
        self.objs = objs

        self.CreateLut()

    def CreateLut(self):
        self.lut = np.zeros([256,3], dtype=np.uint8)
        for obj in self.objDict['objects']: # Load RGB colors as BGR
            self.lut[obj['trainId']][0] = obj['color'][2]
            self.lut[obj['trainId']][1] = obj['color'][1]
            self.lut[obj['trainId']][2] = obj['color'][0]
        self.lut = self.lut.astype(np.float) * 1/255. # scale colors 0-1
        self.lut[self.objDict['background']] = [1.0,1.0,1.0] # Pass Through

    def drawann(self, imgDef, anns):
        annimg = np.zeros(shape=[imgDef['height'], imgDef['width']], dtype=np.uint8)
        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.objDict["classes"]:
                if type(ann['segmentation']) is list:
                    for i in range(len(ann['segmentation'])):
                        contour = np.rint(np.reshape(ann['segmentation'][i], (-1, 2))).astype(np.int32)
                        cv2.drawContours(image=annimg, contours=[contour], contourIdx=-1, color=obj['trainId'] , thickness=cv2.FILLED)
                elif type(ann['segmentation']) is dict:
                    rle = ann['segmentation']
                    compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
                    annmask = mask.decode(compressed_rle)
                    annimg[annmask] = obj['trainId']
                else:
                    print('unexpected segmentation')
            else:
                print('trainId {} >= classes {}'.format(obj['trainId'], self.objDict["classes"]))    
        return annimg

    def __iter__(self):
        self.i = 0
        return self

    def classes(self, anns):
        class_vector = np.zeros(self.objDict['classes'], dtype=np.float32)

        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.objDict["classes"]:
                class_vector[obj['trainId']] = 1.0

        return class_vector

    def DecodeImage(self, bucket, objectname):
        img = None
        imgbuff = self.s3.GetObject(bucket, objectname)
        if imgbuff:
            imgbuff = np.frombuffer(imgbuff, dtype='uint8')
            img = cv2.imdecode(imgbuff, flags=self.imflags)
        return img


    def len(self):
        return len(self.dataset['images'])

    def __next__(self):
        if self.i < self.len():
            result = self.__getitem__(self.i)
            self.i += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.len():
            img_entry = self.dataset['images'][idx]
            imgFile = '{}/{}{}'.format(self.image_paths,self.name_deccoration,img_entry['file_name'])
            img = self.DecodeImage(self.bucket, imgFile)
            ann_entry = self.imgToAnns[img_entry['id']]
            ann = self.drawann(img_entry, ann_entry)
            classes = self.classes(ann_entry)

            result = {'img':img, 'ann':ann, 'classes':classes}

            return result
        else:
            print('CocoStore.__getitem__ idx {} invalid.  Must be >=0 and < CocoStore.len={}'.format(idx, self.len()))
            return None


    def ColorizeAnnotation(self, ann):
        annrgb = [cv2.LUT(ann, self.lut[:, i]) for i in range(3)]
        annrgb = np.dstack(annrgb) 
        return annrgb

    def MergeIman(self, img, ann):
        ann = self.ColorizeAnnotation(ann)
        img = (img*ann).astype(np.uint8)
        return img

class CocoDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, height=480, width=640, imflags=cv2.IMREAD_COLOR, transform=None, target_transform=None, name_deccoration=''):
        self.transform = transform
        self.target_transform = target_transform
        self.height = height
        self.width = width
        self.imflags = imflags

        self.coco = CocoStore(s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=self.imflags, name_deccoration=name_deccoration)


    # Expect img.shape[0]==ann.shape[0] and ann.shape[0]==ann.shape[0]
    def random_resize_crop_or_pad(self, img, ann, target_height, target_width, borderType=cv2.BORDER_CONSTANT, borderValue=0):

        height = img.shape[0]
        width = img.shape[1]
        
        # Pad
        pad = False
        top=0
        bottom=0
        left=0
        right=0
        if target_height > height:
            bottom = int((target_height-height)/2)
            top = target_height-height-bottom
            pad = True
        if target_width > width:
            right = int((target_width-width)/2)
            left = target_width-width-right
            pad = True

        if pad:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, borderValue)
            ann = cv2.copyMakeBorder(ann, top, bottom, left, right, borderType, None, borderValue)

        # Crop
        height = img.shape[0]
        width = img.shape[1]
        maxX = width - target_width
        maxY = height - target_height

        if maxX > 0 or maxY > 0:
            startX = np.random.randint(0, maxX)
            startY = np.random.randint(0, maxY)

            img = img[startY:startY+target_height, startX:startX+target_width]
            ann = ann[startY:startY+target_height, startX:startX+target_width]

        return img, ann

    def __len__(self):
        return self.coco.len()

    def __getitem__(self, idx):
        result = self.coco.__getitem__(idx)
        if result is not None:
            image = result['img']
            label = result['ann']
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)

            if self.width is not None and self.height is not None:
                image, label = self.random_resize_crop_or_pad(image, label,  self.height, self.width)
        else:
            image=None
            label=None
            print('CocoDataset.__getitem__ idx {} returned result=None.'.format(idx))
        return image, label

def Test(args):

    creds = ReadDictJson(args.credentails)
    if not creds:
        print('Failed to load credentials file {}. Exiting'.format(args.credentails))
        return False
    s3def = creds['s3'][0]
    s3 = Connect(s3def)

    if args.test_iterator:
        coco = CocoStore(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.image_path, args.class_dict, imflags=args.imflags)
        for i, iman in enumerate(coco):
            img = coco.MergeIman(iman['img'], iman['ann'])
            cv2.imwrite('cocostoreiterator{:03d}.png'.format(i),img)
            if i >= args.num_images:
                print ('test_iterator complete')
                break

    if args.test_dataset:
        dataset = CocoDataset(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.image_path, args.class_dict, args.height, args.width, imflags=args.imflags)

        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        i = 0
        while i < args.num_images:
            train_features, train_labels = next(iter(train_dataloader))
            j = 0
            while j < args.batch_size and i < args.num_images:
                img = dataset.coco.MergeIman(train_features[j].numpy(), train_labels[j].numpy())
                cv2.imwrite('cocostoredataset{:03d}.png'.format(i),img)
                i += 1
                j += 1
        print ('test_dataset complete')

    print('Test complete')

#objdict = json.load(open('/data/git/mllib/datasets/coco.json'))
#Test(objdict, '/store/Datasets/coco/instances_val2017.json', '/store/Datasets/coco/val2014', 'COCO_val2014_')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-dataset', type=str, default='data/coco/annotations/instances_train2017.json', help='Coco dataset instance json file.')
    parser.add_argument('-image_path', type=str, default='data/coco/train2017', help='Coco image path for dataset.')
    parser.add_argument('-class_dict', type=str, default='model/segmin/coco.json', help='Model class definition file.')
    parser.add_argument('-training', type=str, default='segmin', help='Credentials file.')
    parser.add_argument('-num_images', type=int, default=10, help='Maximum number of images to display')
    parser.add_argument('-batch_size', type=int, default=4, help='Maximum number of images to display')
    parser.add_argument('-test_iterator', type=bool, default=False, help='Maximum number of images to display')
    parser.add_argument('-test_dataset', type=bool, default=True, help='Maximum number of images to display')
    parser.add_argument('-height', type=int, default=640, help='Batch image height')
    parser.add_argument('-width', type=int, default=640, help='Batch image width')
    parser.add_argument('-imflags', type=int, default=cv2.IMREAD_COLOR, help='cv2.imdecode flags')


    args = parser.parse_args()
    return args
    
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

