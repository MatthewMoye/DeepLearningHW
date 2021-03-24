# Deep Learning Homework 2

This directory contains all files necessary for Homework 2.

## Dependencies
Python Version 3.8.3

Pytorch Version 1.8.0

## Dataset
MLDS_hw2_1_data: https://drive.google.com/file/d/1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ/view

## Training the Model
The Model has already been trained and is located in the resources folder. Training the model took about an hour. If you wish to retrain the model, you can do so in either of the following two ways:
```
1. python model_seq2seq.py PATH_TO_FEATURES PATH_TO_LABEL/training_label.json train
2. bash hw2_seq2seq_train.sh PATH_TO_FEATURES PATH_TO_LABEL/training_label.json
```

Examples:
```
1. python model_seq2seq.py MLDS_hw2_1_data/training_data/feat/ MLDS_hw2_1_data/training_label.json train
2. bash hw2_seq2seq_train.sh MLDS_hw2_1_data/training_data/feat/ MLDS_hw2_1_data/training_label.json
```

## Testing the Model
Two ways to test:
```
1. python model_seq2seq.py PATH_TO_FEATURES output_testset.txt test
2. bash hw2_seq2seq_train.sh PATH_TO_FEATURES output_testset.txt
```

Examples:
```
1. python model_seq2seq.py MLDS_hw2_1_data/testing_data/feat/ output_testset.txt test
2. bash hw2_seq2seq_train.sh MLDS_hw2_1_data/testing_data/feat/ output_testset.txt
```

## Bleu Evaluation
```
1. python bleu_eval.py PATH_TO_LABEL/training_label.json output_testset.txt
```

Example:
```
1. python bleu_eval.py MLDS_hw2_1_data/testing_label.json output_testset.txt
```