# Networks

[U-Net](1611.09326.pdf) may be the most widely implemented segmentation network at this time.  The concept is simple and the origonal paper is clear.  Today, the encoder portion of the model that downsizes the image has been replaced with a higher performance CNN.  In my implementaion, I have chosen [Mobilenet V2](Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) implemented in [Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) because of its good performance and small model size.  In addition, its inclusion of inverted residuals and linear bottlenecks present a greater challenging for model conversions such as ONNX and TensorRT and I wanted to stress them a a little.

[FCN](1411.4038.pdf) has generally fallen out of favor as segmentation architecture of choice in the literature today.  Those looking for simplicity have chosen U-Net and those looking for performance have adopted many of the features of DeepLabv3 (little in common wiht V1 and V2 other than the development team).. My interested in FCN is the encoder continues to the point that the image is a 1x1 pixel with a large feature space.  After this, the decoder expands the image back to the origonal size.  At the point that we have converted the image to features, can features of different images (e.g. simeese network) be compared and edited in a useful way?  


[Deeplab V3](1802.02611.pdf)

[Efficientnet](https://arxiv.org/pdf/1905.11946.pdf)
[Source](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)




# Notes:

