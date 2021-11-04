# Selected Topics in Visual Recognition using Deep Learning Homework-1

This repository is the implementation of homework1 of VRDL course.The summary report in both PDF and Markdown foramt could be found in [report directory](./report/).

## Getting started

To install requirements:

```setup
pip install -r requirements.txt
```

Some information and settings of homework dataset: [2021VRDL_HW1_Datasets](https://drive.google.com/drive/folders/1_Rse7MY17IyGIzh8MSuBYpsj3wTVIVRT?usp=sharing)

## Training

To train the basic model (e.g. ResNet) in this repository, run this command:

```train
cd ./basic
python train_model.py
```

To train the fine-grained model (PMG-Progressive-Multi-Granularity-Training) in this repository, run this command:

```train
cd ./PMG-Progressive-Multi-Granularity-Training
python train.py
```
The code of PMG model if modified from [this github repo](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training).

After running the command lines, model will be trained and weights should be saved in correspoding directory.

Currently setting the model hyperparameters directly using command line is not suported, you could modify the argmuents directly in the code if needed.

## Inference

To evaluate the trained model, run:

```eval
python inference.py
```

## Pre-trained Models

You can download pretrained models here:

- [Basic ResNet models](https://drive.google.com/drive/folders/1wQNTbyEp4MGtue5klYhk41QmEhSw3abW?usp=sharing)
- [PMG models](https://drive.google.com/drive/folders/1KpC4Ckc-aoZBG2vZbc7rMr_rYhkxaFhG?usp=sharing)


## Results
Our model achieves the following performance on :
### [CUB 200 2010](http://www.vision.caltech.edu/visipedia/CUB-200.html)

| Model | Feature | Accuracy (on validation set) |
| -------- | -------- | -------- |
| ResNet34     | --     | 62.64%     |
| ResNet50     | --     | 68.38%     |
| ResNet50     | Auto Augment     | 65.28%     |
| ResNet50     | PMG     | 84.57%     |
| ResNet101     | PMG     | 85.30%     |
| Ensemble     | PMG50+101     | 86.12%     |
