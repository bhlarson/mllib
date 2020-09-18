# Embedded Machine Learning Library (mllib)

<p> The embedded machine learning library is a workspace to explore embedding computer vision 
machine learning algorithms.  It's scope is the full supervised machine learning workflow - 
(acquire, annotate, train, test, optimize, deploy, validate).  mllib is
based on cloud-native arctitecture.  

<p>mllib is currently a sandbox to explore ideas and techniques.  As such it is a useful location 
to experament with new techniques.  It is not a stable repository with consistent 
interfaces.  It explores deploying machine learning in constrained environments.

<p>The mllib toolset includes:  Tensorflow 2, Keras, Visual Studio Code, 
Docker, Kubeflow, Kubernetes, Jupyter

## Repository Structure
- [datasets](./datasets/README.md): dataset processing algorithms
- [networks](./networks/README.md): contains a set of convolutional nural networks (CNN) used on various computer vision algorithms
- classify: algorithms to create trained classification networks
- [segment](./segment/README.md): algorithms to create trained segmentation networks
- [serve](./serve/README.md): software deployed to embedded targets.
- [target](./target/README.md): instructions to prepare and target Jetson, Corel, and Raspberry Py boards


[Visual Stuido Code C++ setup](https://code.visualstudio.com/docs/cpp/config-linux)
[Developing in Containers](https://code.visualstudio.com/docs/remote/containers)
[Create a development container](https://code.visualstudio.com/docs/remote/create-dev-container)
[Example remote C++ Debugging](https://github.com/tttapa/VSCode-Docker-Cpp)

# To Do:
- Remove dead code
- Consistant document at all levels
- Import Embedded Classification to classify
- Instructions to setup development environment
- Instructions to use mllib
- Jupyter examples
- Include images in documentation