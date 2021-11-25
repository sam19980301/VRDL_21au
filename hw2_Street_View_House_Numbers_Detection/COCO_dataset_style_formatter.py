import os
import h5py
import numpy as np
import cv2
import json

# Setting Paths
path = r'./train/train/'
file = os.path.join(path, 'digitStruct.mat')
f = h5py.File(file)
names = f['digitStruct/name']
bboxs = f['digitStruct/bbox']

# Extraction Function


def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)


bbox_prop = ['height', 'left', 'top', 'width', 'label']


def get_img_boxes(f, idx=0):
    meta = {key: [] for key in bbox_prop}

    box = f[bboxs[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))
    return meta


def gen_coco_format_data():
    obj_count = 0
    images = list()
    annotations = list()

    # for idx in range(100):
    for idx in range(names.size):
        if (idx % 5000 == 0):
            print(idx)
        img_name = get_img_name(f, idx=idx)
        img_box = get_img_boxes(f, idx=idx)
        height, width = cv2.imread(os.path.join(path, img_name)).shape[:2]
        images.append(dict(
            id=idx,
            file_name=img_name,
            height=height,
            width=width))

        for height, left, top, width, label in zip(
                img_box['height'], img_box['left'], img_box['top'], img_box['width'], img_box['label']):
            # poly = [(top,left),(top+height,left),(top+height,left+width),(top,left+width)]
            poly = [(left, top), (left + width, top),
                    (left + width, top + height), (left, top + height)]
            poly = [float(p) for x in poly for p in x]
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=label,
                bbox=[left, top, width, height],
                area=height * width,
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': i, 'name': str(i if i % 10 else 0)} for i in range(1, 11)])

    return coco_format_json


if __name__ == '__main__':
    coco_format_json = gen_images_and_annotations()
    with open("SVHN_train.json", "w") as outfile:
        json.dump(coco_format_json, outfile)
