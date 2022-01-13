# Selected Topics in Visual Recognition using Deep Learning Homework-4

This repository is the implementation of homework4 of VRDL course. The project compares and analyzes the performance of SwinIR on a customized dataset with 291 trainig images. The summary report in both PDF and Markdown foramt could be found in [report directory](./report/).

## Installation

In this project, framework [KAIR:Image Restoration Toolbox](https://github.com/cszn/KAIR) is used and modified. The repository should be installed first to implement further analysis.

```
git clone https://github.com/cszn/KAIR.git
```

Also `main_test_swinir.py` and `train_swinir_sr_classical_customized.json` should be placed to corresponding path.

## Dataset Download
The [dataset](https://drive.google.com/file/d/1vZH70ai2hj7uIADSm1XQqqMI8N4ue1m2/view?usp=sharing) used in this project is a customized dataset composed of 291 high-resolution training images and 14 low-resolution testing images.

## Model Configurations & Training
Import the model configurations `train_swinir_sr_classical_customized.json` to train the models. Run the following commands:

```
python main_train_psnr.py --opt options/swinir/train_swinir_sr_classical_customized.json
```

The weights of the trained model above is saved in Drive: [Trained Model](https://drive.google.com/file/d/1vwL1xyrlNpLLS8Qf1HChS6TDVb5Yj9-U/view?usp=sharing)

## Inference & Submitting answers

To inference the testing images using trained model amd submit the answer to codalab, run this to generate folder with high-resolution images.

```
python main_test_swinir.py --task classical_sr --scale 3 --training_patch_size 48 --model_path <model_path> --folder_lq <folder_path>
```

## Results
| Model | PSNR |
| ----- | ---- |
| SinwIR| 28.0681 |
