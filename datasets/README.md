# Datasets

To prepare a COCO dataset record directory for training, run getcoco.py to download and expands the coco dataset in your target dataset directory: 
```consol
$ ./dr
# py datasets/getcoco.py /store/Dataset/coco
```
In this example, the output directory is "/store/datasets/coco".  Choose an output directory that can store 60GB data.  Downloading the coco dataset is time consuming.  The consol messages will provide you a time estimate based on your download and extraction speeds.

While COCO is downloading and extracting, extamine the class file "./datasets/coco.json".  The class file is a valid json file with a top-level dictionary defining the "background"  index, an "ignore" index, the number of classes and an array of objects as illustrated in "coco.json".  "coco.json" remaps the 182 COCO names to 4 trainId values (void, person, vehicle, and animal).  "coco.json" color assigns a category color for visualations.  Select the class file used to generate the tfrecord dataset using the -classes_file parameter.

Once coco is downloaded and extracted, run cocorecord.py to generate an optimized .tfrecord training set for segmentation training: 
```consol
# py datasets/cocorecord.py -dataset_path /store/Datasets/coco -record_dir /store/ -classes_file datasets/coco.json
```
Choose the -dataset_path that was downloaded in getcoco.py.  -record_dir defines the output record_dir location.  

With a segmentation training set download and converted to a training set, you can begin training a segmentation networkd described in this [README.md](../segment/README.md)

