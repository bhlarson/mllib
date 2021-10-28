# Datasets
For neural network training, dataset python programs load and prepare datasets for machine learning training, and validation.  

## Download Datasets
The [Cityscapes dataset)[https://www.cityscapes-dataset.com/] is used in many [image segmentation algorithms](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes). It contains segmented outdoor images of various German cities.

## COCO Dataset
The [COCO dataset](https://cocodataset.org) provides object classification and segmentation images and annotations.  [Coco Segmentation](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test) provides a wide variety of scenes and objects in images pulled from [Flicker](https://www.flickr.com/).  Although it is commonly used in accademic [segmentation papers](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test), the mean IOU (intersection over union) scores are typically low and does not show image segmentation algorithms in the best light.  I suspect this is due to the variety of scenes and annotation errors in COCO. To prepare the COCO for training, run getcoco.py to download and and store the coco dataset in the s3 object store: 
```consol
$ ./drb  # build development docker environment
$ ./dr  # run docker development environment 
# python3 datasets/getcoco.py
```
Downloading this 60GB data coco dataset is time-consuming.  The console messages will provide you a time estimate based on your download and extraction speeds.

While COCO is downloading and extracting, examine the class file "mllib/datasets/coco.json".  "coco.json" is a dictionary defining the "background" index, an "ignore" index, the "classes" defining the number of classes and an "objects" array mapping dataset indexes to train indexes.  "coco.json" remaps the 182 COCO names to 4 trainId values (void, person, vehicle, and animal).  Each object entry also assigns a category color for visualizations. 

## Tensorflow Datasets
convert datasets to [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) training set to increase trainging and test performance.  It also enables dataset transformations.  For example, cocorecord converts all coco images to a uniform RGB pixel format, converts image annotations into PNG images, and converts the dataset to application class indexes.  It also enables multiple datasets to be combined into one training set.

Once coco is downloaded and extracted, run cocorecord.py in the same docker environment to generate an optimized .tfrecord training set for segmentation training: 
```console
# python3 datasets/cocorecord.py -trainingset_name coco
```
## PyTorch Datasets

[cocostore.py](cocostore.py) defines a Pytorch dataset that can be loaded by torch.utils.data.DataLoader as shown in this code snipt from [network2d.py](../networks/network2d.py)
```python
trainingset = CocoDataset(s3=s3, bucket=s3def['sets']['dataset']['bucket'], 
    dataset_desc=args.trainingset, 
    image_paths=args.train_image_path,
    class_dictionary=class_dictionary, 
    height=args.height, 
    width=args.width, 
    imflags=args.imflags, 
    astype='float32')

trainloader = torch.utils.data.DataLoader(trainingset)
```
In addition to loading COCO images from an S3 object store, CocoDataset provides configurable image/annotation augmentation.


With a segmentation data set download and converted to a training set, training a segmentation network described in this [README.md](../segment/README.md)

