_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
dataset_type = 'COCODataset'

data = dict(
    train=dict(
        img_prefix='../train/train',
        classes=classes,
        ann_file='../SVHN_train.json'),
    test=dict(
        img_prefix='../test/test',
        classes=classes,
        ann_file='../SVHN_test.json'),
    val=dict(
        img_prefix='../test/test',
        classes=classes,
        ann_file='../SVHN_test.json'),
)

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)
    )
)

load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

workflow = [('train', 1)]
