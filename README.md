# Super_TF
Simple framework to construct machine learning models with tensorflow

## Overview
SuperTF was initially conceived as a means to get familiar with Tensorflow by constructing machine learning models and executing the Tensorflow Tutorials

I have expanded SuperTF overtime and now it has a suite of tools to help in:
- Generation of  datasets as tfrecords files (Currently supports Semantic segmentation, Classification and Sequence Generation)
- Rapid Prototyping of Deep learning models
- Network and Data visualization via tensorboard
- Session management for extended training sessions

## Examples
Please refer to the examples for:
- [Classification dataset generation](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/Make_classification_dataset.py)
- [Classification dataset reading](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/Read_classification_dataset.py)
- [Training LeNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/LeNet.py)
- [Training AlexNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Examples/AlexNet.py)
  (Scemantic segmentation examples will be added shortly)


## Included Network Architectures
I’ve added a several neural network architectures:
### Classification:
- [LeNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Lenet.py) - [Gradient based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [AlexNet](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Alexnet.py) - [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Vgg16](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Vgg16.py) - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Vgg19](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Vgg19.py) - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Inception-resnet-v2-paper](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Inception_resnet_v2py.py) - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- [Inception-resnet-v2-published](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Classification/Inception_resnet_v2a.py) - [Improving Inception and Image Classification in TensorFlow](https://research.googleblog.com/2016/08/improving-inception-and-image.html)

### Segmentation:
- [Full-Resolution Residual Network-A](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Segmentation/FRRN_A.py) - [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323)
 

### Custom:
I've edited and added to certain network architectures to fulfill a certain niche or to improve their performance. These networks are:
- [Unet1024](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Segmentation/Unet1024.py) - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  
  Unet1024 is a simple extension of the orginal Unet architecture, the network accepts an image of size 1024 x 1024 and has 7 encoder-decoder pairs.
  
- [Full-Resolution Residual Network-C](https://github.com/Dhruv-Mohan/Super_TF/blob/master/Super_TF/Model_builder/Architecture/Segmentation/FRRN_C.py) - [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323)
 
  FRRN-C is build upon FRRN-A. Here the center Full-Resolution residual block is replaced by densely conected block of dialated convolutions.
 Moreover the Full-Resolution Residual Network is enclosed in an encoder decoder pair with doubles the input and output resolution. 
- P-Net
- F-Net
- Attn-Lstm
  Attn_Lstm is a multilayer Long short term memory network with [BahdanauAttention](https://arxiv.org/abs/1409.0473). Initial state is set via feature vectors extracted from inception-resent-v2a. Used for image to text generation.
 ### Currently working on:
 
  - Impoving Attn_lstm
  - Preparing wrapper to work with both TF and Pytorch as backend 
