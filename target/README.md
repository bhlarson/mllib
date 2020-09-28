# Embedded Target
This section targets machine learning to various targets.  This includes Nvidia Jetson 
AGX Xavier, Nvidia Jetson NX, Corel.io dev board, and Raspberry Pi.  It also includes 
the TensorRT, Triton, and Tensorflow Lite toolsets.

## Nvidia Jetson Development
Machine learning targeting NVIDIA Jetson begins a problem a dataset.  These instructions
begin with the problem of [image segmentation](https://en.wikipedia.org/wiki/Image_segmentation) based on the [COCO dataset](https://cocodataset.org).  Next we need a trained network which we have created with [train.py](../segment/README.md).  "train.py" produces SavedModel output.

### Nvidia Jetson Setup:
- Follow [JetPack 4.4 install instructions](https://developer.nvidia.com/embedded/jetpack)  specific hardware 

- Set docker permissions
```console
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
$ newgrp docker 
```
- install jetson-stats to use resource utilization:
```console
$ sudo apt update
$ sudo apt install python3-pip
$ sudo -H pip3 install -U jetson-stats
$ sudo reboot
$ jtop
```
- Load mllib
```console
$ sudo mkdir /data/git
$ chown blarson /data
$ mkdir /data/git
$ cd /data/git
$ git clone https://github.com/bhlarson/mllib.git
$ cd mllib
```
- In the remote SSH visual stuido code window, open /data/git/mllib.  
- To build and run the tensorrt docker image, in a terminl window, type:
```console
$ cd /data/git/mllib 
$ ./dbtrt # build tensorrt development docker image
$ ./drtrt # run tensorrt development docker image
```

- serve/app.py provides inference on an image stream from a USB web camera.  From the docker consol, run:
```console
py serve/app.py -loadsavedmodel './saved_model/2020-09-04-05-14-30-dl3'
```
- Open a web broswer to http://localhost:5001/.  I have mapped port 5000 within the docker container to 5001 because port 5000 is used commonly by flask and node development.  You should see the webcam image with people ovrlayed with green, vehicles overlayed red, and animals blue.
- Stop the server by typing ctl+c in the docker consol

## TensorRT
TensorRT is a NVIDIA library for optimizing and running machine learning models on NVIDIA GPUs.  This process of TensorRT inference is based on the [TensorRT Developer-Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/).  Begin by building a trained segmentation network as describe in [../segment/README.md](../segment/README.md)
1. Build dockerfile_trt development docker environment with dbtrt script: docker build --pull -f "dockerfile_trt" -t trt:latest "context"
```console
$ ./dbtrt
```
2. run dockerfile_trt development docker environment script wihth drjb script to load docker envirnment.  This dockerfile mounds the mllib directory to /app (the container working directory).  This means that files within mllib are available within the container.  Changes in the containers /var directory will be changed and preserved in the mllib directory without requireing container rebuilds or file copies.
```console
$ ./drtrt
```
3. Run 
description:
- docker run : run docker image
- --gpus '"device=0"' 
- -it : creating an interactive bash shell in the container
- --rm : Automatically remove the container when it exits
- --cap-add=CAP_SYS_ADMIN 
- -v "$(pwd):/app" : map map the current working directory to docker image volume /app
- -v "/store:/store" : map volume /store to docke image volume /store
- -p 5001:5000/tcp : map port 5001 to docker port 5000 for flask server
- -p 3000:3000 : map port 3000 to docker image port 3000 for Visual Studio Code debugging
- --device /dev/video0:/dev/video0 : map /dev/video0 device to docker image as /dev/video0
- trt:latest : docker image to run

3. Use TF-TRT to convert savedmodel to TensorRT Model:
```console
py target/trt.py -savemodel ./saved_model/2020-09-04-05-14-30-dl3
```
3 (alt) Convert from Tensorflow SavedModel to ONNX model to TensorRT Model.  Despite the extra step, this path is recommended in [Speeding up Deep Learning Inference Using TensorFlow, ONNX, and TensorRT](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/) and has support for many neural network structurs.

4.Inference uisng the Pythion TensorRT engine:

```console
py target/inftrt.py
```

### Links
- [Jetson Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html)
- [User Guide](https://developer.download.nvidia.com/assets/embedded/secure/jetson/xavier/docs/nv_jetson_agx_xavier_developer_kit_user_guide.pdf)
- [Jetpack Release Notes](https://docs.nvidia.com/jetson/jetpack/release-notes/index.html)
- [Jetson Downloads](https://developer.nvidia.com/embedded/downloads)
- [Speeding up Deep Learning Inference Using TensorFlow, ONNX, and TensorRT](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)


## Google Corel.io dev board:
https://coral.ai/products/dev-board/
Enable SSH : https://stackoverflow.com/questions/59325078/cannot-connect-to-coral-dev-board-after-updating-to-4-0-mdt-shell-does-not-work#
1) use ssh-keygen to create private and pub key files.
2) append (or copy) the pubkey file to target /home/mendel/.ssh/authorized_keys
3) copy the private key file to ~/.config/mdt/keys/mdt.key
4) add to local .ssh/config to something like this:

Host tpu
         IdentityFile ~/.config/mdt/keys/mdt.key
         IdentitiesOnly=yes


## To start the serial consol:
![Serial Connection](devboa]rd-serial-power-co.jpg)
> sudo screen /dev/ttyUSB0 115200

## ssh over USB
![USB OTG](devboard-power-data-co.jpg)
> mdt shell

## ssh over Ethernet/WIFI
> mdt devices # get network address over OTG connection 
Returns network name and address
eml             (192.168.1.69) 

>  ssh mendel@192.168.1.69

## sftp in Nautilu (Ubuntu file browser)
Add named configuration to ssh;
> gedit ~/.ssh/config

Add configuraiton with specified IP address:

Host eml
	 HostName 192.168.1.69         
	 IdentityFile ~/.config/mdt/keys/mdt.key
         IdentitiesOnly=yes
         User mendel

![nautilus sftp connection](sftp_nautilus.png)


## sftp filezilla
> filezilla

Select File->Site Manager

![filezilla sftp connection](filezilla.png)

Enable SSH : https://stackoverflow.com/questions/59325078/cannot-connect-to-coral-dev-board-after-updating-to-4-0-mdt-shell-does-not-work#
1) use ssh-keygen to create private and pub key files.
2) append (or copy) the pubkey file to target /home/mendel/.ssh/authorized_keys
3) copy the private key file to ~/.config/mdt/keys/mdt.key
4) add to local .ssh/config to something like this:

Host tpu
         IdentityFile ~/.config/mdt/keys/mdt.key
         IdentitiesOnly=yes

Docker build training/test image
> docker build --pull --rm -f "dockerfile" -t ml:latest context
> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 mllib:latest

Jupyter notebook development:
docker pull jupyter/tensorflow-notebook

<ol type="1">
    <li>System Setup</li>
        <ol type="a">
            <li>Ubuntu</li>
                <ol type="i">
                    <li>os</li>
                    <li>ssh</li>
                    <li>file system</li>  
                </ol>
            <li>MicroK8s or Kubernetes</li>
                <ol type="i">
                    <li>docker</li>
                    <li>snap</li>
                    <li>microk8s</li>  
                </ol>
            <li><a href=https://zero-to-jupyterhub.readthedocs.io/en/latest/setup-jupyterhub/index.html>Jupyter Hub</a> </li>
            <li> <a href=https://min.io>MINIO</a> data storage
            <li><a href=https://github.com/opencv/cvat>CVAT</a></li>  
        </ol>
    <li>Collect images for training, test and validation</li>
    <li>Annotation images using</li>
    <li>Convert annotations to TFRecord training set</li>
        <ol type="a">
            <li></li>
        </ol>
    <li>Select segmentation model</li>
    <li>Select inference hardware</li>
    <li>Select inference server</li>
    <li>Train model</li>
    <li>Verify trained model</li>
    <li>Optimize model for inference hardware</li>
    <li>Deploy to inference hardware</li>
    <li>Validate inference results</li>
    <li></li>
    <li></li>
        <ol type="a">
            <li></li>
            <li></li>
            <li></li>
        </ol>
    <li></li>
    <li></li>
</ol>

## TensorRT conversion of Tensorflow saved model
```console
$ 
# py target/trt.py -debug
```

# Notes:
- 1. Development docker image
   > docker run --device=/dev/video0:/dev/video0 --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store:/store" -p 8889:8888/tcp -p 8009:8008/tcp -p 5001:5000/tcp -p 3001:3000 ml:latest