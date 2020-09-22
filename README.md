# Embedded Machine Learning Library (mllib)

<p> This embedded machine learning library is a workspace to explore embedding computer vision 
machine learning algorithms.  Its scope is the full supervised machine learning workflow - 
(acquire, annotate, train, test, optimize, deploy, validate).  mllib is
based on cloud-native arctitecture. Below, illustrates the mllib image segmetation with the unet network of the coco dataset.  The left image illustrates the validation set segmentation and right the trained network segmentation.

![Image Segmentation](img/ann-pred4.png)

<p>mllib is currently a sandbox to explore ideas and techniques.  As such, it is a useful location 
to experiment with new techniques.  It is not a stable repository with consistent interfaces.

<p>The mllib toolset includes:  Tensorflow 2, Keras, Jupyter, TensorRT, TensorflowLite, Visual Studio Code, Docker, Kubeflow, Kubernetes,  

## Repository Structure
mllib directories define specific steps to perform supervised machine learning.  The README.md within these subdirectories describe how to perform each step.  This include:
- [datasets](./datasets/README.md): dataset processing algorithms
- [networks](./networks/README.md): contains a set of convolutional nural networks (CNN) used on various computer vision algorithms
- classify: algorithms to create trained classification networks
- [segment](./segment/README.md): algorithms to create trained segmentation networks
- [target](./target/README.md): instructions to prepare and target PC, Jetson, Corel, and Raspberry Py boards
- [serve](./serve/README.md): model inference on target platforms

### Development Environment Setup
- Begin with a deep learning compuer (e.g. [lambda workstation](https://lambdalabs.com/)) or cloud GPU server.  A good development GPUs now is [Titan RTX](https://www.nvidia.com/en-us/deep-learning-ai/products/titan-rtx/) with 24GB memory for big models and batch sizes.  A rule of thumb I have followed is 2x system meomory to GPU memory ( == lots of memory).  In addition to a 2GB SSD for the OS and programs, include enough non-volatile storage to store big datastes.  My preference is a 10TB 3.5" HDD for lots of storage at a moderate cost.
- A USB webcam is used in in mllib as an image source for some algorithms.  The [Logitech C920](https://www.logitech.com/en-us/product/hd-pro-webcam-c920) is an expensive webcam supported in Windows, Linux, Jetson, Corel.io and Raspberry Pi through OpenCV.
- On the Development PC: Install Ubuntu 18.04, [Visual Studio Code](https://code.visualstudio.com/), the lateest [NVIDIA drivers](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal), [docker](https://www.docker.com/products/docker-desktop), and the [NVIDIA docker extension](https://github.com/NVIDIA/nvidia-docker )
- In Visual Studio Code, install RemoteSSH extension by Microsoft
- Load the mllib project.  From the command prompt:
```console
sudo mkdir /data
sudo chown $USER /data
mkdir /data/git
cd /data/git
git https://github.com/bhlarson/mllib.git
```

## References:
[Visual Stuido Code C++ setup](https://code.visualstudio.com/docs/cpp/config-linux)
[Developing in Containers](https://code.visualstudio.com/docs/remote/containers)
[Create a development container](https://code.visualstudio.com/docs/remote/create-dev-container)
[Example remote C++ Debugging](https://github.com/tttapa/VSCode-Docker-Cpp)
[Debugging C++ Programs Remotely With SSH Access Using Visual Studio Code](https://medium.com/@shyabithdickwella/debugging-c-programs-remotely-with-ssh-access-using-visual-studio-code-6fe4582b1bf9)

## To Do:
- Document at all levels
- Import Embedded Classification to classify
- Instructions to setup development environment
- Instructions to use mllib
- Jupyter examples
- Include images in documentation