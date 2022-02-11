import numpy as np
import cv2
import random

class RandomResize():
    def __init__(self, 
        height=640, 
        width=640, 
        imflags=cv2.IMREAD_COLOR, 
        name_decoration='',
        normalize=True, # Normalize image to mean of 0 and std of 1
        enable_transform=True, 
        add_noise=True,
        flipX=True,
        flipY=False,
        rotate=15, 
        scale_min=0.75, 
        scale_max=1.25, 
        offset=0.1,
        astype='float32',
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
        noise_type='gauss',
        noise_mean=0,
        noise_variance=0.1,
        saltvspapper=0.5,
        saltvspapper_prob=0.01,
    ):
        self.height = height
        self.width = width
        self.imflags = imflags

        self.normalize = normalize
        self.enable_transform = enable_transform
        self.add_noise = add_noise
        self.flipX = flipX
        self.flipY = flipY
        self.rotate = rotate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset = offset
        self.astype = astype
        self.borderType=borderType
        self.borderValue=borderValue
        self.noise_type = noise_type
        self.noise_mean=noise_mean
        self.noise_variance=noise_variance,
        self.saltvspapper=saltvspapper
        self.saltvspapper_prob=saltvspapper_prob

    '''Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.'''


    def noisy(self, noise_typ,image):
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = self.noise_mean
            var = self.noise_variance
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
            
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = self.saltvspapper
            amount = self.saltvspapper_prob
            noisy = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            noisy[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            noisy[coords] = 0
  
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)

        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss

        else:
            noisy = image

        return noisy


    def resize_im(self, img, ann=None):
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
        if self.height > height:
            bottom = int((self.height-height)/2)
            top = self.height-height-bottom
            pad = True
        if self.width > width:
            right = int((self.width-width)/2)
            left = self.width-width-right
            pad = True

        if pad:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, self.borderType, None, self.borderValue)
            if ann is not None:
                ann = cv2.copyMakeBorder(ann, top, bottom, left, right, self.borderType, None, self.borderValue)

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
                if ann is not None:
                    ann = cv2.warpAffine(src=ann, M=mat, dsize=(width, height))

        # Crop
        height = img.shape[0]
        width = img.shape[1]
        maxX = width - self.width
        maxY = height - self.height

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
            img = img[startY:startY+self.height, startX:startX+self.width]
            if ann is not None:
                ann = ann[startY:startY+self.height, startX:startX+self.width]

        if self.add_noise:
            img = self.noisy(self.noise_type, img)

        return img, imgMean, imgStd, ann