# Deep Learning Homework 3

This directory contains all files necessary for Homework 3.

## Dependencies
Python Version 3.7

Pytorch Version 1.8.0

## Dataset
CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html

## Training the Model
To train a model

## Generate Images
To generate images

## FID Score
Fid score was computed using https://github.com/mseitzer/pytorch-fid.

To recompute the fid score first install the package above
```
pip install pytorch-fid
```

To compute fid score
```
python -m pytorch_fid PATH_TO_RESULTS\MODEL_TO_EVAL\images_fake PATH_TO_RESULTS\MODEL_TO_EVAL\images_real
```
Example
```
python -m pytorch_fid results\DCGAN\images_fake results\DCGAN\images_real
```

## Results
The model achieved a fid score of 
