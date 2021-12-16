# Some basic setup:
# Setup detectron2 logger
import random
import cv2
import json
import os
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries

# import some common detectron2 utilities

# if your dataset is in COCO format, this cell can be replaced by the
# following three lines:
register_coco_instances(
    "nuclei_train",
    {},
    "nuclei_train.json",
    "coco_dataset/train/")

dataset_dicts = DatasetCatalog.get('nuclei_train')
dataset_metadata = MetadataCatalog.get("nuclei_train")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("nuclei_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1  # 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 24 * 500
cfg.SOLVER.STEPS = [24 * 400, 24 * 450]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
# NOTE: this config means the number of classes, but a few popular
# unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = './X101_E500_anchor'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
