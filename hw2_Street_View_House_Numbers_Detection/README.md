# Selected Topics in Visual Recognition using Deep Learning Homework-2

This repository is the implementation of homework2 of VRDL course. The project compares and analyzes the performance of Faster RCNN, Cascade RCNN, YOLOv3 and Deformable DETR on SVHN dataset. The summary report in both PDF and Markdown foramt could be found in [report directory](./report/).

## Installation

In this project, framework [MMDetection](https://github.com/open-mmlab/mmdetection) is used. The package should be installed first to implement further analysis. Refer to [MMDetection official installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) for more information.

```
# Create a conda virtual environment and activate it.
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

# Install PyTorch and torchvision following the official instructions
conda install pytorch torchvision -c pytorch

# Install MMDetection
pip install openmim
mim install mmdet
```

## Dataset Download & Preprocess
The [dataset](https://drive.google.com/drive/folders/19LQtYNQaqdTwfzbtIypHXWOYrXfi2EeW?usp=sharing) used in this project is a subset of [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/), the amout of training images and the format of label information is a bit different. To transfer labels into [Coco dataset style](https://cocodataset.org/#format-data), run the following command:

```
python COCO_dataset_style_formatter.py
```

## Model Configurations
Finally, import the model configurations into MMDetection `configs` directory to train the models. Run the following commands:

```
cd my_confgis

mv faster_rcnn/faster_rcnn_r50_fpn_1x_coco_SVHN.py ../mmdetection/configs/faster_rcnn
mv cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SVHN.py ../mmdetection/configs/cascade_rcnn
mv yolo/yolov3_d53_mstrain-416_273e_coco_SVHN.py ../mmdetection/configs/yolo
mv yolo/yolov3_d53_mstrain-608_273e_coco_SVHN.py ../mmdetection/configs/yolo
mv deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SVHN.py ../mmdetection/configs/deformable_detr
```

Some pretrained models is used, so we need to download them:

```
cd mmdetection
mkdir checkpoints
cd checkpoints

wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
wget https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth
wget https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth
wget https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth
```

## Training

To train the 4 models, (Faster-RCNN, Cascade-RCNN, Yolov3, Deformable DETR) run this command:

```train
cd mmdetection
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_SVHN.py
python tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SVHN.py
python tools/train.py configs/yolo/yolov3_d53_mstrain-416_273e_coco_SVHN.py
python tools/train.py configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SVHN.py
```

The weights of the models above is saved in Drive: [Trained Models](https://drive.google.com/drive/folders/1KfcGQ5EQSm6eZsDoZp28G_f2a2KlwTTB?usp=sharing)

## Inference

To inference the testing images using trained model, run this to generate bbox file in `.json` format

```
cd mmdetection

./tools/dist_test.sh \
    /work_dirs/cascade_rcnn_r50_fpn_1x_coco_SVHN/cascade_rcnn_r50_fpn_1x_coco_SVHN.py \
    /work_dirs/cascade_rcnn_r50_fpn_1x_coco_SVHN/epoch_12.pth \
    1 \
    --format-only \
    --options "jsonfile_prefix=./cascade_rnn_inference"

./tools/dist_test.sh \
    /work_dirs/faster_rcnn_r50_fpn_1x_coco_SVHN/faster_rcnn_r50_fpn_1x_coco_SVHN.py \
    /work_dirs/faster_rcnn_r50_fpn_1x_coco_SVHN/epoch_12.pth \
    1 \
    --format-only \
    --options "jsonfile_prefix=./faster_rcnn_inference"

./tools/dist_test.sh \
    /work_dirs/yolov3_d53_mstrain-416_273e_coco_SVHN/yolov3_d53_mstrain-416_273e_coco_SVHN.py \
    /work_dirs/yolov3_d53_mstrain-416_273e_coco_SVHN/epoch_273.pth \
    1 \
    --format-only \
    --options "jsonfile_prefix=./yolov3_inference"

./tools/dist_test.sh \
    /content/drive/MyDrive/Course/VRDL_2021au/HW2/work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_coco_SVHN/deformable_detr_twostage_refine_r50_16x2_50e_coco_SVHN.py \
    /content/drive/MyDrive/Course/VRDL_2021au/HW2/work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_coco_SVHN/epoch_25.pth \
    1 \
    --format-only \
    --options "jsonfile_prefix=./deformable_detr_inference"
```

To submit the answer to codalab, run `generate_answer.py` to generate the results.
More information could be found via this [Colab Jupyter Notebook](https://colab.research.google.com/drive/1GhV3ZyXSyvNHupWSeOnB09k3Hx_i0Gqw?usp=sharing).

## Results
We summarize the mAP and inference speed of all the models mentioned above, and run time is evaluated on Nvidia Tesla K80 GPU offered by Google Colab. Inference speed may fluctuate according to the enviornment setting of Colab, hence relative speed may be a better metric.

| Model | mAP (%) | Inference Speed (ms) | Relative Speed |
| -------- | -------- | -------- | -------- |
| Faster RCNN | 39.29 | 308.3 | 1x |
| Cascade RCNN | 41.12 | 434.7 | 0.71x |
| Yolov3 | 37.10 | 75.5 | 4.08x |
| Deformable DETR | 36.59 | 351.4 | 0.88x |
