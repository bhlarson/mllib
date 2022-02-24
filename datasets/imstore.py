import sys
import os
from pycocotools import mask
import numpy as np
import cv2
import json
from pathlib import Path, PurePath
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(''))
from utils.s3 import s3store, Connect
from utils.jsonutil import ReadDictJson
from utils.imutil import ImUtil

class ImagesStore:

    def __init__(self, s3, bucket, dataset_desc, class_dictionary):

        self.s3 = s3
        self.bucket = bucket
        self.dataset_desc = s3.GetDict(bucket,dataset_desc)
        self.class_dictionary = s3.GetDict(bucket,class_dictionary) 

        self.imflags = cv2.IMREAD_COLOR 
        if self.dataset_desc is not None and 'image_colorspace' in self.dataset_desc:
            if self.isGrayscale(self.dataset_desc['image_colorspace']):
                self.imflags = cv2.IMREAD_GRAYSCALE
        self.anflags = cv2.IMREAD_GRAYSCALE 

        self.bare_image = self.dataset_desc['image_pattern'].replace('*','')
        self.bare_label = self.dataset_desc['label_pattern'].replace('*','')

        self.images = []
        self.labels = []
        
        self.CreateIndex()
        self.CreateLut()
        
        self.i = 0

    def isGrayscale(self, color_str):
        if color_str.lower() == 'grayscale':
            return True
        return False

    def ImagenameFromLabelname(self, lbl_filename):
        return lbl_filename.replace(self.bare_label, self.bare_image)


    def CreateIndex(self):
        file_list = self.s3.ListObjects( self.dataset_desc['bucket'], setname=self.dataset_desc['prefix'], pattern=None, recursive=self.dataset_desc['recursive'])
        imagedict = {}
        if self.dataset_desc['image_path']==self.dataset_desc['label_path']:
            for im_filename in file_list:
                if PurePath(im_filename).match(self.dataset_desc['image_pattern']):
                    imagedict[im_filename] = None
            for lbl_filename in file_list:
                if PurePath(lbl_filename).match(self.dataset_desc['label_pattern']):
                    im_filename = self.ImagenameFromLabelname(lbl_filename)
                    if im_filename in imagedict:
                        imagedict[im_filename] = lbl_filename

        for key in imagedict:
            if imagedict[key] is not None:
                self.images.append(key)
                self.labels.append(imagedict[key])

    def CreateLut(self):
        self.lut = np.zeros([256,3], dtype=np.uint8)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            self.lut[obj['trainId']][0] = obj['color'][2]
            self.lut[obj['trainId']][1] = obj['color'][1]
            self.lut[obj['trainId']][2] = obj['color'][0]
        self.lut = self.lut.astype(float) * 1/255. # scale colors 0-1
        self.lut[self.class_dictionary ['background']] = [1.0,1.0,1.0] # Pass Through

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

    def DecodeImage(self, bucket, objectname, flags):
        img = None
        numTries = 3
        for i in range(numTries):
            imgbuff = self.s3.GetObject(bucket, objectname)
            if imgbuff:
                imgbuff = np.frombuffer(imgbuff, dtype='uint8')
                img = cv2.imdecode(imgbuff, flags=self.imflags)
            if img is None:
                print('ImagesStore::DecodeImage failed to load {}/{} try {}'.format(bucket, objectname, i))
            else:
                break
        return img

    def ConvertLabels(self, ann):
        trainAnn = np.zeros_like(ann)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            if not (obj['id'] == obj['trainId'] and obj['id'] == 0):
                trainAnn[ann==obj['id']] = obj['trainId']
        return trainAnn

    def len(self):
        return len(self.images)

    def __next__(self):
        if self.i < self.len():
            result = self.__getitem__(self.i)
            self.i += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.len():
            img = self.DecodeImage(self.bucket, self.images[idx], self.imflags)
            ann = self.DecodeImage(self.bucket, self.labels[idx], self.anflags)
            ann = self.ConvertLabels(ann)
            result = {'img':img, 'ann':ann}

            return result
        else:
            print('LitStore.__getitem__ idx {} invalid.  Must be >=0 and < LitStore.len={}'.format(idx, self.len()))
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

        if(self.isGrayscale(self.dataset_desc['image_colorspace'])):
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        img = (img*ann).astype(np.uint8)
        return img

    def DisplayImAn(self, img, ann, seg, mean, stdev):

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        iman = self.MergeIman(img, ann, mean, stdev)
        imseg = self.MergeIman(img, seg, mean, stdev)

        iman = cv2.putText(iman, 'Segmentation',(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)
        imseg = cv2.putText(imseg, 'TensorRT',(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)

        im = cv2.hconcat([iman, imseg])
        return im

class ImagesDataset(Dataset):
    def __init__(self, s3, bucket, dataset_desc, class_dictionary, 
        height=640, 
        width=640, 
        image_transform=None,
        label_transform=None,
        normalize=True, 
        enable_transform=True, 
        flipX=True, 
        flipY=False, 
        rotate=3, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        astype='float32'
    ):
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.height = height
        self.width = width

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset
        self.astype = astype

        self.store = ImagesStore(s3, bucket, dataset_desc, class_dictionary)


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
        return self.store.len()

    def __getitem__(self, idx):
        result = self.store.__getitem__(idx)
        if result is not None and result['img'] is not None and result['ann'] is not None:
            image = result['img']
            label = result['ann']

            if self.width is not None and self.height is not None:
                image, label, imgMean, imgStd = self.random_resize_crop_or_pad(image, label,  self.height, self.width)

            if image is not None and label is not None:
                if len(image.shape) < 3:
                    image = np.expand_dims(image, axis=-1)

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
            print('ImagesDataset.__getitem__ idx {} returned result=None.'.format(idx))
        return image, label, imgMean, imgStd

def Test(args):

    s3, creds, s3def = Connect(args.credentails)

    if args.test_iterator:
        store = ImagesStore(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.class_dict)
        for i, iman in enumerate(store):
            img = store.MergeIman(iman['img'], iman['ann'])
            cv2.imwrite('cocostoreiterator{:03d}.png'.format(i),img)
            if i >= args.num_images:
                print ('test_iterator complete')
                break

    if args.test_dataset:
        dataset = ImagesDataset(s3, s3def['sets']['dataset']['bucket'], args.dataset, args.class_dict)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        i = 0
        while i < args.num_images:
            images, train_labels, train_mean, train_stdev = next(iter(dataloader))
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = np.squeeze(images)
            j = 0
            while j < args.batch_size:
                train_labels[j].cpu().numpy()
                img = dataset.store.MergeIman(images[j], train_labels[j].cpu().numpy(), train_mean[j].item(), train_stdev[j].item())
                cv2.imwrite('cocostoredataset{:03d}{:03d}.png'.format(i,j),img)
                j += 1
            i += 1
        print ('test_dataset complete')

    print('Test complete')

#objdict = json.load(open('/data/git/mllib/datasets/coco.json'))
#Test(objdict, '/store/Datasets/coco/instances_val2017.json', '/store/Datasets/coco/val2014', 'COCO_val2014_')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-dataset', type=str, default='annotations/lit/dataset.yaml', help='Image dataset file')
    parser.add_argument('-class_dict', type=str, default='model/crisplit/lit.json', help='Model class definition file.')
    parser.add_argument('-training', type=str, default='crisplit', help='Credentials file.')
    parser.add_argument('-num_images', type=int, default=10, help='Maximum number of images to display')
    parser.add_argument('-batch_size', type=int, default=4, help='Maximum number of images to display')
    parser.add_argument('-test_iterator', type=bool, default=False, help='Maximum number of images to display')
    parser.add_argument('-test_dataset', type=bool, default=True, help='Maximum number of images to display')

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    import argparse
    args = parse_arguments()

    if args.debug:
        print("Wait for debugger attach")
        import debugpy

        debugpy.listen(address=('0.0.0.0', args.debug_port))
        debugpy.wait_for_client()  # Pause the program until a remote debugger is attached
        print("Debugger attached")

    Test(args)

