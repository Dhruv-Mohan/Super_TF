# Super_TF
Simple framework to construct machine learning models with tensorflow

## Overview
SuperTF was initially conceived as a means to get familiar with Tensorflow by constructing machine learning models and executing the Tensorflow Tutorials

I have expanded SuperTF overtime and now it has a suite of tools to help in:
- Generation of  datasets as tf.records files (Currently supports Semantic segmentation and Classification)
- Rapid Prototyping of Deep learning models
- Network and Data visualization via tensorboard
- Session management for extended training sessions

## Examples
Please refer to the examples for:
- [Classification dataset generation](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/Make_classification_dataset.py)
- [Classification dataset reading](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/Read_classification_dataset.py)
- [Training LeNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/LeNet.py)
- [Training AlexNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/AlexNet.py)


## Included Network Architectures
I’ve added a several neural network architectures:
### Classification:
- [LeNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Lenet.py)
- [AlexNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Alexnet.py)
- [Vgg16](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Vgg16.py)
- [Vgg19](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Vgg19.py)
- [Inception-resnet-v2-paper](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Inception_resnet_v2py.py)
- [Inception-resnet-v2-published](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Inception_resnet_v2a.py)

### Segmentation:
- [Full-Resolution Residual Network-A](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Segmentation/FRRN_A.py)

### Custom:
I've edited and added to certain network architectures to fullfill a certain niche or to improve their performance these networks are:
- [Unet1024](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Segmentation/Unet1024.py)
  
  Unet1024 is a simple extension of the orginal Unet architecture, the network accepcts an image of size 1024 x 1024 and has 7 pairs of ecoder-decoder pairs
- [Full-Resolution Residual Network-C](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Segmentation/FRRN_C.py)
 
  FRRN-C is build upon FRRN-A. Here the center Full-Resolution residual block is replaced by densely conected block of dialated convolutions.
 Moreoever the Full-Resolution Residual Network is enclosed in an ecoder decoder pair with doubles the input and output resolution. 
