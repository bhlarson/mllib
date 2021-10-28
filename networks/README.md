# Networks

# Image Segmentation
The [U-Net](1611.09326.pdf) segmentation convolutional neural network (CNN) concept is simple and the paper is well written.  In [unet.py](unet.py) the encoder portion of the model that downsizes the image has been replaced with [Mobilenet V2](Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) because of its good performance and small size.  I implement this in [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).  My purpose of this implementation is to evaluate the TensorRT model optimization.  Mobilenet V2's inclusion of inverted residuals and linear bottlenecks may present a greater challenging for model conversions to ONNX and TensorRT then a strictly convolutional network

[FCN](1411.4038.pdf) has generally fallen out of favor as segmentation architecture of choice in the literature today.  Those looking for simplicity have chosen U-Net and those looking for performance have adopted many of the features of DeepLabv3 (little in common with V1 and V2 other than the development team). My interested in FCN is the encoder continues to the point that the image is a 1x1 pixel with a large feature space.  After this, the decoder expands the image back to the original size.  At the point that we have converted the image to features, can features of different images (e.g. Siamese network) be compared and edited in a useful way?  


[Deeplab V3](1802.02611.pdf)

[Efficientnet](https://arxiv.org/pdf/1905.11946.pdf)
[Source](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Neural Architecture Search
[cell2d.py](cell2d.py) is my neural architecture search experiment inspired by [DARTS](https://arxiv.org/pdf/1806.09055.pdf) and [LEAStereo](https://github.com/XuelianCheng/LEAStereo).  My objective is not to find a novel architecture but to reduce model size during training weight optimization.  During training, cell2d simultaneously improves network accuracy, reduces channels and decreases convolution depth.  "cell2d" train and test on the CIFAR10 dataset are part of the default "Test" function in cell2d: 
```console
./dtb # build the Pytorch development docker environment
./dtr # run the Pytorch development docker environment
py networks/cell2d.py # run the training & evaluation network test
```
[network2d.py](network2d.py) builds cell2d into a U-NET where neural architecture search minimizes segmentation cross entropy loss and minimizes network size by reducing the number of layer channels, reducing convolutional depth, and reducing UNET depth.  To prepare the segmentation dataset, first prepare the [COCO Dataset](../datasets/README.md) preparation.  Next, build the network2d as follows
```console
./dtb # build the Pytorch development docker environment
./dtr # run the Pytorch development docker environment
py networks/network2d.py # run the training & evaluation network test
```


```console
tensorboard --logdir /store/test/nassegtb --bind_all
```

# Notes:

