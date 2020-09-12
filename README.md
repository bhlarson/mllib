# Embedded Machine Learning Library (mllib)

<p> The embedded machine learning library is a workspace to explore embedding computer vision 
machine learning algorithms.  It's scope is the full supervised machine learning workflow - 
(acquiring, annotating, training, testing, optimizing, deploying, validation).  mllib is
based on cloud-native arctitecture througout.  

<p>mllib is currently a sandbox to explore ideas and techniques.  As such it is a useful location 
to experament with new techniques.  It is not currently a stable repository with consistent 
interfaces.  As sush, it will hopefully help others to explore deploying machine learning in
constrained environments.

<p>The toolset used to devlop and deploy mllib includes:  Tensorflow 2, Keras, Visual Studio Code, 
Docker, Kubeflow, Kubernetes, Jupyter

## Repository Structure
- datasets: dataset processing algorithms
- networks: contains a set of convolutional nural networks (CNN) used on various computer vision algorithms
- classify: algorithms to create trained classification networks
- segment: algorithms to create trained segmentation networks
- serve: software deployed to embedded targets.
- target: instructions to prepare and target Jetson, Corel, and Raspberry Py boards


# To Do:
- Remove dead code
- Consistant document at all levels
- Import Embedded Classification to classify
- Instructions to setup development environment
- Instructions to use mllib
- Jupyter examples
- Include images in documentation