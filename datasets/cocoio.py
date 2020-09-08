from pycocotools import mask
import numpy as np
import cv2
import json
from collections import defaultdict

class CocoIO:
    def __init__(self, objDict, cocoJson, imagePaths, name_deccoration = None):
        self.objDict = objDict
        self.imagePaths = imagePaths
        self.dataset = json.load(open(cocoJson))
        self.createIndex()
        self.i = 0
        self.name_deccoration = name_deccoration


    def createIndex(self):
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

    def __next__(self):
        if self.i < self.len():
            img = self.dataset['images'][self.i]
            imgFile = '{}/{}{}'.format(self.imagePaths,self.name_deccoration,img['file_name'])
            ann = self.imgToAnns[img['id']]
            annImg = self.drawann(img, ann)
            result = {'img':imgFile, 'ann':annImg}

            self.i += 1
            return result
        else:
            raise StopIteration

    def len(self):
        return len(self.dataset['images'])

def Test(objdict, cocojson, imgPath, name_deccoration = None):
    coco = CocoIO(objdict,cocojson, imgPath, name_deccoration)

    lut = np.zeros([256,3], dtype=np.uint8)
    for obj in objdict['objects']: # Load RGB colors as BGR
        lut[obj['trainId']][0] = obj['color'][2]
        lut[obj['trainId']][1] = obj['color'][1]
        lut[obj['trainId']][2] = obj['color'][0]
    lut = lut.astype(np.float) * 1/255. # scale colors 0-1
    lut[objdict['background']] = [1.0,1.0,1.0] # Pass Through

    for iman in coco:
        img = cv2.imread(iman['img'])
        ann = [cv2.LUT(iman['ann'], lut[:, i]) for i in range(3)]
        ann = np.dstack(ann) 
        iman = (img*ann).astype(np.uint8)
        cv2.imshow( "Display window", iman); 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#objdict = json.load(open('/data/git/mllib/datasets/coco.json'))
#Test(objdict, '/store/Datasets/coco/instances_val2017.json', '/store/Datasets/coco/val2014', 'COCO_val2014_')       