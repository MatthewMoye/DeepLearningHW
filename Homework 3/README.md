# Deep Learning Homework 3

This directory contains all files necessary for Homework 3.

## Dependencies
Python Version 3.7

Pytorch Version 1.8.0

## Dataset
CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html

## Training the Model
To train each model
```
python .\main.py --model ACGAN --evaluate train
python .\main.py --model DCGAN --evaluate train
python .\main.py --model WGAN --evaluate train
```

## Generate Images
To generate images for each model
```
python .\main.py --model ACGAN --evaluate generate
python .\main.py --model DCGAN --evaluate generate
python .\main.py --model WGAN --evaluate generate
```

## FID Score
Fid score was computed using https://github.com/mseitzer/pytorch-fid.

To compute the fid score first install the package
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
Example using GPU
```
python -m pytorch_fid results\DCGAN\images_fake results\DCGAN\images_real --gpu 0
```

## Results
The DCGAN achieved a FID score of 51.4

The WGAN-GP achieved a FID score of 48.9

The ACGAN achieved a FID score of 47.8
