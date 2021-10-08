# Embedded Machine Learning Library (mllib)

<p> This embedded machine learning library is a workspace to explore embedding computer vision 
machine learning algorithms.  Its scope is the full supervised machine learning workflow - 
(acquire, annotate, train, test, optimize, deploy, validate).  mllib employs a microservices arctitecture. 

Mllib image segmetation using the unet network and trained using the coco dataset is shown below.  The left image illustrates a human segmented validation image and right shows the results of Tensorflow training and conversion to TensorRT Float16 inference on Jetson NX.  This project explores the process of creating and transforming models to performed machine learned impage processing on embedded hardware.

![Image Segmentation](img/segtrt4355000.png)



mllib is currently a sandbox to explore ideas and techniques.  As such, it is a useful location 
to experiment with new techniques.  It is not a stable repository with consistent interfaces.

The mllib toolset includes:  Tensorflow 2, Keras, Jupyter, TensorRT, TensorflowLite, Visual Studio Code, Docker, airflow, and Kubernetes.  The target embedded hardware includes [Jetson AGX](https://developer.nvidia.com/embedded/jetson-agx-xavier), [Jetson NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit), [Google Corel](https://coral.ai/products/dev-board/), and [Raspberry Pi](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
</br>

<img src="img/jetsonAGX.jpeg" alt="JetsonAGX" height="175"/>
<img src="img/jetsonNX.jpeg" alt="JetsonNX" height="175"/>
<img src="img/googleCorel.jpeg" alt="Corel" height="175"/>
<img src="img/raspberryPi.png" alt="Corel" height="175"/>

## Repository Structure
mllib directories define specific steps to perform supervised machine learning.  The README.md within these subdirectories describe how to perform each step.  This include:
- [datasets](./datasets/README.md): dataset processing algorithms
- [networks](./networks/README.md): convolutional nural networks (CNN) used by other image processing algorithms
- [classify](./classify/README.md): algorithms to train and test classification networks
- [segment](./segment/README.md): algorithms to train and test segmentation networks
- [target](./target/README.md): instructions in scripts to prepare and target PC, Jetson, Corel, and Raspberry Pi boards
- [serve](./serve/README.md): model inference on target platforms
- utils: shared utility libraries

## Process
Embedded machine learned image processing is an emerging field.  Although science fiction and its close cousin, press headlines, build the impression that the area is stable and mature, this is far from the case.  Consiquently, the process I recommend is very quick development cycles moving algorithms quickly from devlopment to test on the target platform.  In this past, moving algorithms from a development environment to embedded hardware would have involved a complete rewrite of software in an new runtime environment, all of these target systems run linux, can execute pythion algorithms in vitualized docker environments with hardware access to powerful machine learning coprocessors.  Consiquently, my embedded machine learning process is to create an algorithm and with little or no training or optimization, move it and test it in the embedded environment.  This verifies the entire tool chain can handle the model structure.  Once any targeting problems are rectivied, model training and optimization is improved and target performance is verified.

To target Jetson boards, I am using the [Tensorflow->ONNX->TensorRT](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/) path.  For Corel and Raspberry Pi, I am following the TensorFlow->[Tensorflow Lite](https://www.tensorflow.org/lite/microcontrollers) path.

## Development Environment
The basics of a embedded ML imaging environment include a development workstation, network, embedded device and webcam.
</br>
<img src="img/Workstation.jpeg" alt="Workstation" height="175"/>
<img src="img/switch90.jpeg" alt="switch" height="175"/>
<img src="img/jetsonNX.jpeg" alt="JetsonNX" height="175"/>
<img src="img/webcam.jpeg" alt="Logitech c920 " height="175"/>
</br>

- I prefer a deep learning workstation (e.g. [lambda workstation](https://lambdalabs.com/)) rather than cloud for development.  This will host your development environment, datasets, and provide objet storage to share data between the development and embedded environments.  This can be built up from a current workstation or purchased as a complete sytem.  
- The key component of a deep learning workstaiton is the GPU where training and inference is performed.  If there were not a global semiconductior shortage, a variety of gaming to professional graphics cards are available at a wide price point. At this writing [Titan RTX](https://www.nvidia.com/en-us/deep-learning-ai/products/titan-rtx/) with 24GB memory for big models and batch sizes would be a good choice.  This is a moving target and will take some research.
- After GPU, system memory is my biggest bottleneck to keeping the GPU working efficiently.  A rule of thumb I have followed is 2x system meomory to GPU memory ( e.g. 48 GB RAM for a 24GB Titan RTX memory).  
- Next is storage for big datasets.  My preference is a 10TB 3.5" HDD for lots of storage at a moderate cost in addition to an NVME drive for runtime cashing.
- What about CPUs?   Choose one that enables the fastes PCIe and memory speed and can keep up with the non GPU pre and post processing.  I typically choose a lower-cost CPU that maximizes my communication speeds.
- A USB webcam is a flexiable and fun image source.  I learn a lot by interacting live with ML algorithms that I wouldn't on saved image sets (low light, high contrast, saturation, etc). I use the lhe [Logitech C920](https://www.logitech.com/en-us/product/hd-pro-webcam-c920) because it is supported in Windows, Linux, Jetson, Corel.io and Raspberry Pi through OpenCV.
- [Ubuntu](https://ubuntu.com) linux distribution
- [Visual Studio Code](https://code.visualstudio.com/) is a great free development environment
- [Python](https://www.python.org/) is the primarly language for ML development and of this project
- [Docker](https://www.docker.com/) defines and runs the runtime environment for developing and targeting embedded environments.  All code in this project is run within a docker environment.
- [MINIO](https://min.io/) s3 object storage stores and distributes machine learning data between embedded devices, servers, and develelopment PCs.  
- [Jetson NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit) is a capabile target platform for machine learned image processing if you are choosing a target platform.  


# Setup
On the development workstation:
- Setup [Ubuntu desktop](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)
- Install the current [nvidia drivers](https://linuxize.com/post/how-to-nvidia-drivers-on-ubuntu-20-04/)
- Install [docker](https://docs.docker.com/engine/install/ubuntu/)
- Install [microk8s kubernetes](https://microk8s.io/docs)
```console
sudo snap install microk8s --classic
sudo snap install microk8s --channel=1.22.2/stable --classic
sudo usermod -a -G microk8s $USER
sudo chown -f -R $USER ~/.kube
su - $USER
microk8s status --wait-ready
microk8s enable dns gpu helm3 storage registry
```
- Install [Visual Studio Code](https://code.visualstudio.com/)
- In Visual Studio Code, install Python Remote Development, Jupyter, Json, and Getlens extensions
- and the [NVIDIA docker extension](https://github.com/NVIDIA/nvidia-docker )
- Create a [minio object storage](https://docs.min.io/docs/minio-quickstart-guide.html)


- Load the mllib project.  From the command prompt:
```console
sudo mkdir /data
sudo chown $USER /data
mkdir /data/git
cd /data/git
git https://github.com/bhlarson/mllib.git
```
## Set-up secure minio repository
- Let's Encrypt wildcard certificae
- Kubernetes secret: [Generate TLS Secret for kubernetes](https://software.danielwatrous.com/generate-tls-secret-for-kubernetes/)
- Base 64 encode certificate
```console
cat cert.pem | base64 | awk 'BEGIN{ORS="";} {print}' > tls.crt
cat privkey.pem | base64 | awk 'BEGIN{ORS="";} {print}' > tls.key
```
- create a credentials file mllib/creds.json defining S3 access crediantials.  It should have the the strucure below.  Replace the "<>" values with the values of your object storage
```json
{
    "s3":[
        {"name":"mllib-s3", 
            "type":"trainer", "address":"<s3 url>", 
            "access key":"<s3 access key>", 
            "secret key":"<s3 secret key>",
            "tls":true, 
            "cert_verify":false,
            "cert_path": null,
            "sets":{
                "dataset":{"bucket":"mllib","prefix":"data", "dataset_filter":"" },
                "trainingset":{"bucket":"mllib","prefix":"training", "dataset_filter":"" },
                "model":{"bucket":"mllib","prefix":"model", "dataset_filter":"dl3" },
                "test":{"bucket":"mllib","prefix":"test", "dataset_filter":"" }
            } 
         }
    ]
}
```

# Results
## Performance
For 480 height, 512 width images, the following table shows the UNET accuracy, similarity, and inference time:
|Software|Hardware|Images|Accuracy|Similarity|Inference time(s)|
|---|---|---|---|---|---|
|Tensorflow Foat32|X86-64 RTX 6000|5000.0|0.947432|0.668267|0.076956|
|Onnx Foat32|X86-64 RTX 6000|5000.0|0.947474|0.667503|0.153856|
|TensorRT Foat16|X86-64 RTX 6000|5000.0|0.947155|0.667532|0.008323|
|Tensorflow Foat32|Jetson AGX|5000.0|0.945176|0.668743|0.231636|
|TensorRT Foat16|Jetson AGX|5000.0|0.945007|0.665993|0.029665|
|Tensorflow Foat32|Jetson NX|5000.0|0.941661|0.666916|0.370289|
|TensorRT Foat16|Jetson NX|5000.0|0.946283|0.668803|0.046575|
<br />

### Confusion Matrix - Tensorflow Float 32 - X86-64 RTX6000
<img src="img/Tensorflow-X86-RTX6000.png"/>
</br>

### Confusion Matrix - TensorRT Float 16 - Jetson AGX
<img src="img/TensorRT-JetsonAGX.png"/>
</br>

### Confusion Matrix - TensorRT Float 16 - Jetson NX
<img src="img/TensorRT-JetsonNX.png"/>
</br>


## To Do:
- Import Embedded Classification to classify
- Instructions to setup development environment
- Instructions to use mllib
- Jupyter examples

## Notes:
Setup Microk8s snap install microk8s
Setup Kubectl snap install kubectl [Working with kubectl](https://microk8s.io/docs/working-with-kubectl)

## References:
- [Visual Stuido Code C++ setup](https://code.visualstudio.com/docs/cpp/config-linux)
- [Developing in Containers](https://code.visualstudio.com/docs/remote/containers)
- [Create a development container](https://code.visualstudio.com/docs/remote/create-dev-container)
- [Example remote C++ Debugging](https://github.com/tttapa/VSCode-Docker-Cpp)
- [Debugging C++ Programs Remotely With SSH Access Using Visual Studio Code](https://medium.com/@shyabithdickwella/debugging-c-programs-remotely-with-ssh-access-using-visual-studio-code-6fe4582b1bf9)

