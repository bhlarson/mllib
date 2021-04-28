# Datasets
The dataset python programs convert datasets to [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) trainingset to increase trainging and test performance.  It also enables dataset transformations.  For example, cocorecord converts all coco images to a uniform RGB pixel format, rasters image annotations into PNG images, and converts the dataset to application class indexes.  It also enables multiple datasets to be combined into one training set.

To prepare the COCO for training, run getcoco.py to download and and store the coco dataset in the s3 object store: 
```consol
$ ./drb  # build development docker environment
$ ./dr  # run docker development environment 
# python3 datasets/getcoco.py
```
Downloading this 60GB data coco dataset is time consuming.  The consol messages will provide you a time estimate based on your download and extraction speeds.

While COCO is downloading and extracting, examine the class file "mllib/datasets/coco.json".  "coco.json" is a dictionary defining the "background"  index, an "ignore" index, the "classes" defining the number of classes and an "objects" array mapping dataset indexes to train indexes.  "coco.json" remaps the 182 COCO names to 4 trainId values (void, person, vehicle, and animal).  Each "object entry also assigns a category color for visualations.  

Once coco is downloaded and extracted, run cocorecord.py in the same docker envorinment to generate an optimized .tfrecord training set for segmentation training: 
```consol
# python3 datasets/cocorecord.py -trainingset_name coco
```

With a segmentation data set download and converted to a training set, training a segmentation network described in this [README.md](../segment/README.md)

