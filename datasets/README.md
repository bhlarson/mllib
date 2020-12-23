# Datasets
The dataset python programs convert datasets to [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) trainingset.  This conversion inproves trainging and test performance.  It also enables dataset transformations.  For example, cocorecord converts all coco images to RGB pixel format, rasters image annotations into PNG images, and converts to application class indexes.  It enables multiple datasets to be combined into one training set.

To prepare the COCO for training, run getcoco.py to download and and store the coco dataset in the s3 object store: 
```consol
$ ./drb  # build development docker environment
$ ./dr  # run docker development environment 
# py datasets/getcoco.py /store/Dataset/coco
```
Downloading the 60GB data coco dataset is time consuming.  The consol messages will provide you a time estimate based on your download and extraction speeds.

While COCO is downloading and extracting, examine the class file "./datasets/coco.json".  The class file is a valid json file with a top-level dictionary defining the "background"  index, an "ignore" index, the number of classes and an array of objects as illustrated in "coco.json".  "coco.json" remaps the 182 COCO names to 4 trainId values (void, person, vehicle, and animal).  "coco.json" color assigns a category color for visualations.  Select the class file used to generate the tfrecord dataset using the -classes_file parameter.

Once coco is downloaded and extracted, run cocorecord.py to generate an optimized .tfrecord training set for segmentation training: 
```consol
# py datasets/cocorecord.py
```

With a segmentation data set download and converted to a training set, you can begin training a segmentation network described in this [README.md](../segment/README.md)

