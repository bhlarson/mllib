# Datasets

<p> Datasets scripts coverts datasets to a tfrecord trainig set.  Dataset scripts include:

- cocorecord.py: converts coco datast to a tfrecord segementation training set.  From the mllib directory:

```consol
$ dr # load development docker environment
# dataset/cocorecord.py -classes_file dataset/coco.json
```