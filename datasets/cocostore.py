import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pymlutil.s3 import s3store, Connect
from pymlutil.jsonutil import ReadDictJson

def resize_crop_or_pad(img, target_height, target_width, 
    normalize=True, borderType=cv2.BORDER_CONSTANT, borderValue=0, 
    astype='float32'):

    imgMean = None
    imgStd = None
    imgtype = img.dtype.name
    if normalize:
        imgMean = np.mean(img)
        imgStd = np.std(img)
        img = (img - imgMean)/imgStd
    
    if astype is not None:
        img = img.astype(astype)
    elif img.dtype.name is not  imgtype:
        img = img.astype(imgtype)

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

    # Crop
    height = img.shape[0]
    width = img.shape[1]
    maxX = width - target_width
    maxY = height - target_height

    crop = False
    startX = 0
    startY = 0
    if maxX > 0:
        startX = int(maxX/2)
        crop = True
    if  maxY > 0:
        startY = int(maxY/2)
        crop = True
    if crop:

        img = img[startY:startY+target_height, startX:startX+target_width]

    return img, imgMean, imgStd

class CocoStore:

    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=cv2.IMREAD_COLOR, name_decoration='' ):

        self.s3 = s3
        self.bucket = bucket
        self.dataset_desc = dataset_desc
        self.class_dictionary = class_dictionary
        self.image_paths = image_paths
        self.name_decoration = name_decoration
        self.imflags = imflags

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

        if self.class_dictionary  is not None:
            for obj in self.class_dictionary ['objects']:
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
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            self.lut[obj['trainId']][0] = obj['color'][2]
            self.lut[obj['trainId']][1] = obj['color'][1]
            self.lut[obj['trainId']][2] = obj['color'][0]
        self.lut = self.lut.astype(np.float) * 1/255. # scale colors 0-1
        self.lut[self.class_dictionary ['background']] = [1.0,1.0,1.0] # Pass Through

    def drawann(self, imgDef, anns):
        annimg = np.zeros(shape=[imgDef['height'], imgDef['width']], dtype=np.uint8)
        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
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
                print('trainId {} >= classes {}'.format(obj['trainId'], self.class_dictionary ["classes"]))    
        return annimg

    def __iter__(self):
        self.i = 0
        return self

    def classes(self, anns):
        class_vector = np.zeros(self.class_dictionary ['classes'], dtype=np.float32)

        for ann in anns:
            obj = self.catToObj[ann['category_id']]
            if obj['trainId'] < self.class_dictionary ["classes"]:
                class_vector[obj['trainId']] = 1.0

        return class_vector

    def DecodeImage(self, bucket, objectname):
        img = None
        numTries = 3
        for i in range(numTries):
            imgbuff = self.s3.GetObject(bucket, objectname)
            if imgbuff:
                imgbuff = np.frombuffer(imgbuff, dtype='uint8')
                img = cv2.imdecode(imgbuff, flags=self.imflags)
            if img is None:
                print('CocoStore::DecodeImage failed to load {}/{} try {}'.format(bucket, objectname, i))
            else:
                break
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
            imgFile = '{}/{}{}'.format(self.image_paths,self.name_decoration,img_entry['file_name'])
            img = self.DecodeImage(self.bucket, imgFile)
            ann_entry = self.imgToAnns[img_entry['id']]
            ann = self.drawann(img_entry, ann_entry)
            classes = self.classes(ann_entry)
            result = {'img':img, 'ann':ann, 'classes':classes}

            return result
        else:
            print('CocoStore.__getitem__ idx {} invalid.  Must be >=0 and < CocoStore.len={}'.format(idx, self.len()))
            return None

    # Display functions
    def ColorizeAnnotation(self, ann):
        annrgb = [cv2.LUT(ann, self.lut[:, i]) for i in range(3)]
        annrgb = np.dstack(annrgb) 
        return annrgb

    def MergeIman(self, img, ann, mean=None, stDev = None):
        if mean is not None and stDev is not None:
            img = (img*stDev) + mean

        if self.class_dictionary is not None:
            ann = self.ColorizeAnnotation(ann)
        img = (img*ann).astype(np.uint8)
        return img

    def DisplayImAn(self, img, ann, seg, mean, stdev):

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        iman = self.MergeIman(img, ann, mean, stdev)
        imseg = self.MergeIman(img, seg, mean, stdev)

        iman = cv2.putText(iman, 'Segmentation',(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)
        imseg = cv2.putText(imseg, 'TensorRT',(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)

        im = cv2.hconcat([iman, imseg])
        #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im

class CocoDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, 
        height=640, 
        width=640, 
        imflags=cv2.IMREAD_COLOR, 
        image_transform=None,
        label_transform=None,
        name_decoration='',
        normalize=True, 
        enable_transform=True, 
        flipX=True, 
        flipY=False, 
        rotate=15, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        astype='float32'
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.height = height
        self.width = width
        self.imflags = imflags

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset
        self.astype = astype

        self.coco = CocoStore(s3, bucket, dataset_desc, image_paths, class_dictionary, imflags=self.imflags, name_decoration=name_decoration)


    # Expect img.shape[0]==ann.shape[0] and ann.shape[0]==ann.shape[0]
    def random_resize_crop_or_pad(self, img, ann, target_height, target_width, borderType=cv2.BORDER_CONSTANT, borderValue=0):
        imgMean = None
        imgStd = None
        imgtype = img.dtype.name
        if self.normalize:
            imgMean = np.mean(img)
            imgStd = np.std(img)
            img = (img - imgMean)/imgStd
        
        if self.astype is not None:
            img = img.astype(self.astype)
        elif img.dtype.name is not  imgtype:
            img = img.astype(imgtype)

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

        # Transform
        if self.enable_transform:
                height, width = img.shape[:2]

                matFlip = np.identity(3)
                if self.flipX and np.random.choice(np.array([True, False])):
                    matFlip[0,0] *= -1.0
                    matFlip[0,2] += width-1
                if self.flipY and np.random.choice(np.array([True, False])):
                    matFlip[1,1] *= -1.0
                    matFlip[1,2] += height-1

                scale = np.random.uniform(self.scale_min, self.scale_max)
                angle = np.random.uniform(-self.rotate, self.rotate)
                offsetX = width*np.random.uniform(-self.offset, self.offset)
                offsetY = height*np.random.uniform(-self.offset, self.offset)
                center = (width/2.0 + offsetX, height/2.0 + offsetY)
                matRot = cv2.getRotationMatrix2D(center, angle, scale)
                matRot = np.append(matRot, [[0,0,1]],axis= 0)

                mat = np.matmul(matFlip, matRot)
                mat = mat[0:2]


                img = cv2.warpAffine(src=img, M=mat, dsize=(width, height))
                ann = cv2.warpAffine(src=ann, M=mat, dsize=(width, height))

        # Crop
        height = img.shape[0]
        width = img.shape[1]
        maxX = width - target_width
        maxY = height - target_height

        crop = False
        startX = 0
        startY = 0
        if maxX > 0:
            startX = np.random.randint(0, maxX)
            crop = True
        if  maxY > 0:
            startY = np.random.randint(0, maxY)
            crop = True
        if crop:

            img = img[startY:startY+target_height, startX:startX+target_width]
            ann = ann[startY:startY+target_height, startX:startX+target_width]

        return img, ann, imgMean, imgStd

    def __len__(self):
        return self.coco.len()

    def __getitem__(self, idx):
        result = self.coco.__getitem__(idx)
        if result is not None and result['img'] is not None and result['ann'] is not None:
            image = result['img']
            label = result['ann']

            if self.width is not None and self.height is not None:
                image, label, imgMean, imgStd = self.random_resize_crop_or_pad(image, label,  self.height, self.width)

            image = torch.from_numpy(image).permute(2, 0, 1)
            label = torch.from_numpy(label)

            if self.image_transform:
                image = self.image_transform(image)
            if self.label_transform:
                label = self.label_transform(label)
            
        else:
            image=None
            label=None
            imgMean = None
            imgStd = None
            print('CocoDataset.__getitem__ idx {} returned result=None.'.format(idx))
        return image, label, imgMean, imgStd

def Test(args):

    s3, creds, s3def = Connect(args.credentails)

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
            train_features, train_labels, train_mean, train_stdev = next(iter(train_dataloader))
            j = 0
            while j < args.batch_size and i < args.num_images:
                img = dataset.coco.MergeIman(train_features[j], train_labels[j], train_mean[j], train_stdev[j])
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

