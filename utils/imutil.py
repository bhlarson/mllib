import numpy as np
import cv2

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

class ImUtil():
    def __init__(self, s3, bucket, dataset_desc, image_paths, class_dictionary, 
        height=640, 
        width=640, 
        imflags=cv2.IMREAD_COLOR, 
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

        self.CreateLut()

    def CreateLut(self):
        self.lut = np.zeros([256,3], dtype=np.uint8)
        for obj in self.class_dictionary ['objects']: # Load RGB colors as BGR
            self.lut[obj['trainId']][0] = obj['color'][2]
            self.lut[obj['trainId']][1] = obj['color'][1]
            self.lut[obj['trainId']][2] = obj['color'][0]
        self.lut = self.lut.astype(np.float) * 1/255. # scale colors 0-1
        self.lut[self.class_dictionary ['background']] = [1.0,1.0,1.0] # Pass Through

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