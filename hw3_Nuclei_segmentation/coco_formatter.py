import os
import cv2
import numpy as np
import json
from cocoapi.PythonAPI.pycocotools import mask
from itertools import groupby

# Clean given dataset
for root, dirs, files in os.walk("./dataset", topdown=False):
    for name in dirs:
        if name.startswith('.ipynb_checkpoints'):
            rm_path = os.path.join(root, name)
            print(rm_path)
            os.system(f'rm -rf {rm_path}')

# Make images directory in coco style
dest_dir = 'coco_dataset'
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
dest_train_dir = os.path.join(dest_dir, 'train')
if not os.path.exists(dest_train_dir):
    os.mkdir(dest_train_dir)

train_img_dir = 'dataset/train'
test_img_dir = 'dataset/test'

for img in os.listdir(train_img_dir):
    source_file = os.path.join(train_img_dir, img, 'images', f'{img}.png')
    dest_file = os.path.join(dest_train_dir, f'{img}.png')
    if not os.path.exists(dest_file):
        os.system(f'cp {source_file} {dest_file}')

os.system(f'cp -R {test_img_dir} {dest_dir}')


# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
def polygonFromMask(maskedArr):

    contours, _ = cv2.findContours(
        maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask.frPyObjects(
        segmentation,
        maskedArr.shape[0],
        maskedArr.shape[1])
    RLE = mask.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0]  # , [x, y, w, h], area


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(
            groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


# Make annotations json file
# training set
obj_count = 0
images = list()
annotations = list()

for idx, img_name in enumerate(os.listdir(train_img_dir)):
    height, width = cv2.imread(os.path.join(
        train_img_dir, img_name, 'images', f'{img_name}.png')).shape[:2]
    images.append(dict(
        id=idx,
        file_name=f'{img_name}.png',
        height=height,
        width=width))

    for masks in os.listdir(os.path.join(train_img_dir, img_name, 'masks')):
        mask_img_data = cv2.imread(
            os.path.join(
                train_img_dir,
                img_name,
                'masks',
                masks))[
            :,
            :,
            0]

        polygon = polygonFromMask(mask_img_data)

        rle = binary_mask_to_rle(np.asfortranarray(mask_img_data))
        compressed_rle = mask.frPyObjects(
            rle, rle.get('size')[0], rle.get('size')[1])
        bbox = mask.toBbox(compressed_rle).tolist()
        area = int(mask.area(compressed_rle))

        data_anno = dict(
            image_id=idx,
            id=obj_count,
            category_id=1,
            segmentation=[polygon],
            bbox=bbox,
            area=area,
            iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1
        if (obj_count % 1000 == 0):
            print(obj_count)

coco_format_json = dict(
    images=images,
    annotations=annotations,
    categories=[{"id": 1, "name": '1'}])

with open("nuclei_train.json", "w") as outfile:
    json.dump(coco_format_json, outfile)
