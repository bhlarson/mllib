# Image segmentation

- run python scripts from <project path>/mllib

# Workflow
1. Prpare docker runtime
```console
drb # load and build runtime docker image
djb # load and build jupyter docker image
```
1. Load dataset to train segmentation
```console
dr # loads development docker image
python3 datasets/getcoco.py
```
1. Prepare training set: 
```console
dr # loads development docker image
python3 segment/maketrain.py
```
1. Train:
```console
dr # loads development docker image
python3 segment/train.py
```
1. Test:
```console
dr # loads development docker image
python3 segment/test.py
```
1. View test results:
```console
dj # loads jupyter docker image
```
In browser URL edit box, enter: http://localhost:8888/
1. Validate on target in environment
1. Deploy
1. Analyze peformance
1. Update

