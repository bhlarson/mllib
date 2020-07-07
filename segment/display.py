import copy
import cv2
import numpy as np
import tensorflow as tf

def FindObjectType(index, objectTypes):
    for objectType in objectTypes:
        if objectType['trainId'] == index:
            return objectType
    return None

def DefineFeature(c, iClass, minArea = 0, maxArea = float("inf"), iContour=1):
    M = cv2.moments(c)

    area = M['m00']
    # Ignore contours that are too small or too large
    
    if area < 1.0 or area < minArea or area > maxArea:
        return {}

    center = (M['m10']/M['m00'], M['m01']/M['m00'])

    rect = cv2.boundingRect(c)   

    feature = {'class':iClass, 'iContour':iContour, 'center':center,  'rect': rect, 'area':area, 'contour':c}
    
    return feature

def FilterContours(seg, objectType, config):
    

    [height, width] = seg.shape

    # Area filter
    if 'area_filter_min' in objectType:
        minArea  = objectType['area_filter_min']
    else:
        minArea = config['area_filter_min']
    
    if 'area_filter_max' in objectType:
        maxArea  = objectType['area_filter_max']
    else:
        maxArea = height*width

    iSeg = copy.deepcopy(seg)
    iSeg[iSeg != objectType['trainId']] = 0 # Convert to binary mask        
    
    
    segFeatures = []
    contours, hierarchy = cv2.findContours(iSeg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):

        feature = DefineFeature(c, objectType['trainId'], minArea, maxArea, i)
        if feature:
            segFeatures.append(feature)

    return segFeatures

def ExtractFeatures(seg, objTypes, config):
    segFeatures = []
    for segobj in objTypes:
      if segobj['display']:
        feature_contours = FilterContours(seg, segobj, config)
        segFeatures.extend(feature_contours)


    return segFeatures

def ApplyColors(img, seg, objTypes, configm, alpha = 0.50):
    [height, width] = seg.shape
    img = img.astype(np.float32)
    imgout = copy.deepcopy(img)
    for segobj in objTypes:
      if segobj['display']:
        mask = np.zeros((height,width), dtype=np.float32)
        #mask[seg == segobj['trainId']] = 1 # Convert to binary mask
        mask[seg == segobj['trainId']] = 1.0 # Convert to binary mask
        notmask = cv2.cvtColor(np.logical_not(mask).astype(np.float32), cv2.COLOR_GRAY2BGR)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        imgout = imgout*notmask # zero masked pixels
        #imgout += alpha*img*mask + (1.0-alpha)*mask*np.array(segobj['color']) # color with segmentation color

        #    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #img *= segmask
    #img += (alpha*np.array(segobj['color']))*mask
    img = np.clip(img, 0.0, 255.0)

    return img.astype(np.uint8)

def ColorToBGR(color):
    return (color[2], color[1], color[0])

def DrawFeatures(img, seg, config):
    objTypes = config['trainingset']['classes']['objects']
    features = ExtractFeatures(seg, objTypes, config)
    for feature in features:
        obj = FindObjectType(feature['class'], objTypes)
        if obj and obj['display']:
            cv2.drawContours(img, [feature['contour']], 0, ColorToBGR(obj['color']), thickness=1)

    return img
    #return ApplyColors(img, seg, objTypes, config)



def DrawImAn(img, ann, config):

    img = img.astype(np.uint8)
    ann = ann.astype(np.uint8)

    #annImg = copy.deepcopy(img)
    return DrawFeatures(img, ann, config)


def DrawSeg(img, ann, pred, objTypes, config):
    ann = tf.squeeze(ann)
    pred = tf.squeeze(pred)

    img = img.numpy()
    ann = ann.numpy()
    pred = pred.numpy()

    img = img.astype(np.uint8)
    ann = ann.astype(np.uint8)
    pred = pred.astype(np.uint8)

    annImg = copy.deepcopy(img)
    DrawFeatures(annImg, ann, config)

    predImg = copy.deepcopy(img)
    DrawFeatures(predImg, pred, config)

    return annImg, predImg

# Let's try out the model to see what it predicts before training.
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def MergeImgAn(dataset, config, num=1):
    batch_size = config['batch_size']
    objTypes = config['trainingset']['classes']['objects']
    imgs = []
    i = 0
    for image, mask in dataset.take(num):
      for j in range(batch_size):

        iman = DrawImAn(image[j].numpy(), mask[j].numpy(), config)
        imgs.append(iman)
      i=i+1
    return imgs

def WriteImgAn(dataset, config, num=1, outpath=''):

    imgs = MergeImgAn(dataset, config, num=1)
    for i, img in enumerate(imgs):
        cv2.imwrite('{}/ann-pred{}.png'.format(outpath, i), img)

def CreateImanBounds(img, ann, config):
    ann = create_mask(ann)
    ann = tf.squeeze(ann) # Drop color dimension
    iman = DrawImAn(img, ann.numpy(), config)
    return iman

def CreateIman(img, ann, config):
    ann = create_mask(ann)
    ann = tf.squeeze(ann) # Drop color dimension
    iman = DrawImAn(img, ann.numpy(), config)
    return iman

def CreatePredictions(dataset, model, config, num=1):
    batch_size = config['batch_size']
    objTypes = config['trainingset']['classes']['objects']
    imgs = []
    i = 0
    for image, mask in dataset.take(num):
      for j in range(batch_size):
        pred_mask = create_mask(model.predict(image))

        ann, pred = DrawSeg(image[j], mask[j], pred_mask[j], objTypes, config)
        imgs.append(np.concatenate((ann, pred), axis=1))
      i=i+1
    return imgs

def WritePredictions(dataset, model, config, num=1, outpath=''):

    imgs = CreatePredictions(dataset, model, config, num=1)
    for i, img in enumerate(imgs):
        cv2.imwrite('{}/ann-pred{}.png'.format(outpath, i), img)