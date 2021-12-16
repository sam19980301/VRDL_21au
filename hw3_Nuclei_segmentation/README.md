# Selected Topics in Visual Recognition using Deep Learning Homework-3

This repository is the implementation of homework3 of VRDL course. The project compares and analyzes the performance of Mask RCNN (R50, R101, X101), Cascade Mask RCNN (R50) on Nuclei dataset. The summary report in both PDF and Markdown foramt could be found in [report directory](./report/).

## Installation

In this project, framework [Detectron2](https://github.com/facebookresearch/detectron2) is used. The package should be installed first to implement further analysis. Refer to [Detectron2 official installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for more information.

```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
Also [COCO API](https://github.com/cocodataset/cocoapi) is used to preprocessing the data.

```
git clone https://github.com/cocodataset/cocoapi
cd cocoapi/PythonAPI
make
```

## Dataset Download & Preprocess
The [dataset](https://drive.google.com/file/d/1iaULQi9eWbRORqMtaTfRizC6K3ducg1B/view?usp=sharing) used in this project is a subset of [Nuclei Dataset](https://www.kaggle.com/c/data-science-bowl-2018), the amout of training images and the format of label information is a bit different. To transfer labels into [Coco dataset style](https://cocodataset.org/#format-data), run the following command:

```
python coco_formatter.py
```

## Model Configurations & Training
Finally, import the model configurations `X101_E500_anchor.py` to train the models. Run the following commands:

```
python X101_E500_anchor.py
```

The weights of the trained model above is saved in Drive: [Trained Model](https://drive.google.com/file/d/1Y6VvJ-dvJ0cgZI4MYDpNMjCzkZNpK3jW/view?usp=sharing)

## Inference & Submitting answers

To inference the testing images using trained model amd submit the answer to codalab, run this to generate annotations file in `.json` format

```
python generate_answer.py
```

## Results
| Model | mAP |
| -------- | -------- |
| Mask RCNN R-50| 0.232596 |
| Mask RCNN R-101| 0.227455 |
| Cascade Mask RCNN R50| 0.229755 |
| Mask RCNN X-101 small anchor size| 0.243117 |
