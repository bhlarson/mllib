# Datasets

To prepare a COCO dataset record directory for training, first download the datset.
If needed, first install unzip and set the execute permission on getcoco.sh.  Command line examples are from the <mllib> project directory.
```consol
$ sudo apt install unzip
$ chmod +x ./datasets/getcoco.sh
```

The getcoco.sh script downloads and extracts these archives to the output directory.  In this example, the output directory is "/store/datasets/coco".
```consol
$ ./datasets/getcoco.sh /store/datasets/coco

```
While the datasets are being downloaded and extracted, examine the ./datasets/coco.json file. This remaps the coco segmentation index values to training index values.  In this example, the 182 COCO names are mapped 4 trainId values (void, person, vehicle, and animal).  ./datasets/coco.json also assigns a category color use by segmentation visualations.  Once you have verified mllib functionality, edit the coco.json trainId to include/exclude features of interest.

<p>Once the COCO 2017 datasets are download, "./datasets/cocorecord.py" creates tfrecord trainig set. "dockerfile" provides a container to build, train, and test mllib.  From the mllib directory:

```consol
$ drb # load development docker environment
$ dr # load development docker environment
# dataset/cocorecord.py -classes_file dataset/coco.json
```

```consol
$ dr # load development docker environment
# dataset/cityscapesrecord.py -classes_file dataset/coco.json
```