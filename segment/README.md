# Image segmentation

The mllib/segment folder contains pythion programs to train and test image segmentation in various environments.  This will describe training and testing a UNET segmenter.  [UNET](https://arxiv.org/pdf/1505.04597.pdf)  is a very common fully-convolutional image segmentation network based on a simple structure and well written paper.  

Before training and test, a training set must be be prepared.  The [mllib/dataset/README.md](../dataset/README.md) outlines preparing a trainingset from the COCO dataset.  Please complete this before beginning the process of training a segmentation network.

## UNET Training and Test
1. Prpare and launch the docker runtime.  The following steps are performed witin this docker environment
```console
./db # build the training docker image
./dr # run the training docker image
```
Train the unet segmentation with the COCO training set.  "-trainingset coco" parameter specifies the training will be performed on the coco training set.  "-epochs 10" specificies training will complete after 10 passess through the training set.  "-savedmodelname" specifies the saved model name as "unet"  The model training cross entropy loss is printed in the terminal and training convergence can be observed by comaring loss values over time.

```console
python3 segment/train.py -trainingset coco -savedmodelname unet
```

When training is complete, "testtf.py" runs the trained network agains the valiation set and records the results in tests.json file added to the traingset.   
```console
python3 segment/testtf.py -trainingset coco -model unet
```

The jupyter notebook [mllib/segment/test.ipynb](./segment/test.ipynb) provides test visualizations.  To use this, 
1. Exit the current docker envorinment and load the jupyter docker environment, 
```console
exit # exit the current training docker image
./djpb # build jupyter docker image
./djpr # run jupyter docker image
```
2. Open Jupyter in a browser: [http://localhost:8888](http://localhost:8888)
1. Open the test.ipynb notebook

In browser URL edit box, enter: http://localhost:8888/
1. Validate on target in environment
1. Deploy
1. Analyze peformance
1. Update

