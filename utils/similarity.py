import copy
import numpy as np
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
        if i > 0 and i in iTypes:
            metric = {}
            ta = copy.deepcopy(annotation)
            tb = copy.deepcopy(segmentation)

            ta[ta != i] = 0 # Convert to binary mask
            ta = ta.astype('bool')
            aa = ta.sum()
            metric['annotation area'] = aa.item()


            tb[tb != i] = 0 # Convert to binary mask
            tb = tb.astype('bool')
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
                typeSimilarity[i]['intersection'] += intersection
                typeSimilarity[i]['union'] += union            

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