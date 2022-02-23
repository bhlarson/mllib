import copy
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

def similarity(intersection, union):
    if union > 0:
        similarity = np.array(intersection/float(union)).item()
    elif intersection == 0 and union == 0:
        similarity = 1.0
    else:
        similarity = 0.0

    return similarity

def jaccard(annotation, segmentation, iTypes, typeSimilarity):
    intersections = 0
    unions = 0
    metrics = {}
    unique = np.unique(annotation)
    unique = np.append(unique,np.unique(segmentation))
    unique = np.unique(unique)

    #print(unique)

    for i in unique:
        if i in iTypes:
            metric = {}
            ta = np.zeros_like(annotation, dtype='bool')
            tb = np.zeros_like(segmentation, dtype='bool')

            ta[annotation == i] = 1 # Convert to binary mask
            aa = ta.sum()
            metric['annotation area'] = aa.item()

            tb[segmentation == i] = 1 # Convert to binary mask
            ab = tb.sum()
            metric['segmentation area'] = ab.item()

            andim = ta*tb # Logical AND
            intersection = andim.sum()
            metric['intersection'] = intersection.item()
            intersections += int(intersection)

            orim = ta + tb # Logical OR
            union = orim.sum()
            metric['union'] = union.item()
            unions += int(union)

            if i in typeSimilarity:
                typeSimilarity[i]['intersection'] += int(intersection)
                typeSimilarity[i]['union'] += int(union)

            metric['similarity'] = similarity(intersection, union)
            metrics[iTypes[i]['name']] = metric

    iou = {}
    iou['objects'] = metrics
    iou['image'] = similarity(intersections, unions)
         
    return iou, typeSimilarity, unique

def confusionmatrix(labels, segmentaiton, classes, total_confusion = None):

    confusion = confusion_matrix(labels.flatten(),segmentaiton.flatten(), labels=classes)

    if total_confusion is None:
        total_confusion = confusion
    else:
        total_confusion += confusion
        
    return confusion, total_confusion

def eval_records(fie, objTypes, records, resultsPath):

    results = []
    if not os.path.exists( resultsPath):
        os.makedirs(resultsPath)

    typeSimilarity = {}
    for objType in objTypes:
        typeSimilarity[objType['index']] = {'union':0, 'intersection':0} 

    for id, record in enumerate(records):
        im = cv2.imread(record['image'],cv2.IMREAD_GRAYSCALE)
        an = cv2.imread(record['annotation'],cv2.IMREAD_GRAYSCALE)


        outputs = fie(tf.expand_dims(tf.constant(im),-1))
        seg = np.squeeze(outputs['class_ids'].numpy())

        record['similarity'], typeSimilarity, unique = jaccard(an, seg, iTypes, typeSimilarity)       

        resultsImg = ed.AnnotationPlots(objTypes, cv2.cvtColor(im,cv2.COLOR_GRAY2RGB), an, seg, resultsPath, record)
        record['review'] = resultsImg

        print('similarity: {:.2f}, unique: {} image: {}'.format(record['similarity']['image'], unique, record['image']))
        results.append(record)

    return results, typeSimilarity

def ColorizeAnnotation(ann, lut):
    annrgb = [cv2.LUT(ann, lut[:, i]) for i in range(3)]
    annrgb = np.dstack(annrgb) 
    return annrgb

def MergeIman(img, ann, lut, mean=None, stDev = None):
    if mean is not None and stDev is not None:
        img = (img*stDev) + mean

    ann = ColorizeAnnotation(ann, lut)
    img = (img*ann).astype(np.uint8)
    return img

def MergeImAnSeg(img, ann, seg, lut, mean=None, stdev=None, ann_text='Label', seg_text='Segmentation'):

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    iman = MergeIman(img, ann, lut, mean, stdev)
    imseg = MergeIman(img, seg, lut, mean, stdev)

    iman = cv2.putText(iman, ann_text,(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)
    imseg = cv2.putText(imseg, seg_text,(10,25), font, 1,(255,255,255),1,cv2.LINE_AA)

    iman = cv2.hconcat([iman, imseg])
    return iman

class DatasetResults:
    # imgSave: save image to path defined in imgSave
    # imRecord: recorde image in results
    def __init__(self, class_dictionary, batch_size=1, imStatistics=False, imgSave=None, imRecord=False, task='segmentation'):
        # Prepare datasets for similarity computation
        self.class_dictionary = class_dictionary
        self.batch_size = batch_size
        self.imStatistics = imStatistics
        self.imgSave = imgSave
        self.imRecord = imRecord
        self.task = task

        # Process inputs for evaluation
        self.classSimilarity = {}
        self.objTypes = {}
        for objType in class_dictionary['objects']:
            if objType['trainId'] not in self.objTypes:
                self.objTypes[objType['trainId']] = copy.deepcopy(objType)
                # set name to category for objTypes and id to trainId
                self.objTypes[objType['trainId']]['name'] = objType['category']
                self.objTypes[objType['trainId']]['id'] = objType['trainId']

        for i in self.objTypes:
            self.classSimilarity[i]={'intersection':0, 'union':0}

        self.num_classes = len(self.objTypes)
        self.confusion_labels = range(self.num_classes)
 
        self.CreateLut()

        # Prepaire results data structures
        self.typeSimilarity = {}
        self.images = []
        self.totalConfusion=None
        self.dtSum =0

    def CreateLut(self):
        self.lut = np.zeros([256,3], dtype=np.uint8)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            self.lut[obj['trainId']][0] = obj['color'][2]
            self.lut[obj['trainId']][1] = obj['color'][1]
            self.lut[obj['trainId']][2] = obj['color'][0]
        self.lut = self.lut.astype(np.float) * 1/255. # scale colors 0-1
        self.lut[self.class_dictionary ['background']] = [1.0,1.0,1.0] # Pass Through

    def infer_results(self, iBatch, images, labels, segmentations, mean, stdev, dt):
        self.dtSum += dt
        numImages = len(images)
        dtImage = dt/numImages
        for j in range(numImages):

            result = {'dt':dtImage}

            if self.imgSave is not None or self.imRecord:
                imanseg = MergeImAnSeg(images[j], labels[j], segmentations[j], self.lut, mean[j], stdev[j])
                if self.imgSave is not None:
                    savename = '{}/{}{:04d}.png'.format(self.imgSave, self.task, self.batch_size*iBatch+j)
                    cv2.imwrite(savename, imanseg)
                if self.imRecord:
                    result['image'] = imanseg

            result['similarity'], self.classSimilarity, unique = jaccard(labels[j], segmentations[j], self.objTypes, self.classSimilarity)

            confusion = confusion_matrix(labels[j].flatten(),segmentations[j].flatten(), labels=self.confusion_labels)
            if self.totalConfusion is None:
                self.totalConfusion = confusion
            else:
                self.totalConfusion += confusion

            result['confusion'] = confusion.tolist()

            self.images.append(result)

        return self.totalConfusion

    def Results(self):

        dataset_similarity = {}

        num_images = len(self.images)
        average_time = self.dtSum/num_images
        sumIntersection = 0
        sumUnion = 0
        miou = 0
        for key in self.classSimilarity:
            intersection = self.classSimilarity[key]['intersection']
            sumIntersection += intersection
            union = self.classSimilarity[key]['union']
            sumUnion += union
            class_similarity = similarity(intersection, union)
            miou += class_similarity

            # convert to int from int64 for json.dumps
            dataset_similarity[key] = {'intersection':int(intersection) ,'union':int(union) , 'similarity':class_similarity}

        # miou computation
        positives = np.diagonal(self.totalConfusion)
        total = np.sum(self.totalConfusion,0)+np.sum(self.totalConfusion,1)-positives
        # Remoze zero values
        positives = positives[np.nonzero(total)]
        total = total[np.nonzero(total)]
        iou = positives/total
        miou =np.sum(iou)/self.num_classes

        results = {'confusion':self.totalConfusion.tolist(), 
                   'similarity':dataset_similarity,
                   'average time': average_time,
                   'mean intersection over union': miou,
                   'num images': num_images,
                }
        if self.imStatistics:
            results['images'] = self.images, 

        return results