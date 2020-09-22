# Embedded Machine Learning Library (mllib)

<p> This embedded machine learning library is a workspace to explore embedding computer vision 
machine learning algorithms.  It's scope is the full supervised machine learning workflow - 
(acquire, annotate, train, test, optimize, deploy, validate).  mllib is
based on cloud-native arctitecture.  

<p>mllib is currently a sandbox to explore ideas and techniques.  As such it is a useful location 
to experiment with new techniques.  It is not a stable repository with consistent interfaces.

<p>The mllib toolset includes:  Tensorflow 2, Keras, Jupyter, TensorRT, TensorflowLite, Visual Studio Code, Docker, Kubeflow, Kubernetes,  

## Repository Structure
mllib directories include:
- [datasets](./datasets/README.md): dataset processing algorithms
- [networks](./networks/README.md): contains a set of convolutional nural networks (CNN) used on various computer vision algorithms
- classify: algorithms to create trained classification networks
- [segment](./segment/README.md): algorithms to create trained segmentation networks
- [target](./target/README.md): instructions to prepare and target PC, Jetson, Corel, and Raspberry Py boards
- [serve](./serve/README.md): model inference on target platforms


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