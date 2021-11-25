import os
import numpy as np
import cv2
import json

# Setting Paths
test_path = r'./test/test'
inference_file_name = '../yolov3_inference.bbox.json'


def gen_images():
    images = list()
    annotations = list()

    test_file_list_dir = os.listdir(test_path)
    test_file_list_dir.sort(key=lambda x: int(x[:-4]))

    for idx, img_name in enumerate(test_file_list_dir):
        height, width = cv2.imread(os.path.join(test_path, img_name)).shape[:2]
        images.append(dict(
            id=idx,
            file_name=img_name,
            height=height,
            width=width))

    return images


coco_format_json = dict(
    images=gen_images(),
    annotations=list(),
    categories=[{'id': i, 'name': str(i if i % 10 else 0)} for i in range(1, 11)])


def gen_answer(inference_file_name):
    inference = json.load(open(inference_file_name, 'r'))

    image_map = {img['id']: int(img['file_name'][:-4])
                 for img in coco_format_json['images']}
    categories_map = {cat['id']: int(cat['name'])
                      for cat in coco_format_json['categories']}

    for i in range(len(inference)):
        inference[i]['image_id'] = image_map[inference[i]['image_id']]
        inference[i]['category_id'] = categories_map[inference[i]['category_id']]

    # inference = list(filter(lambda item:item['score']>=0.5, inference))

    with open("answer.json", "w") as outfile:
    json.dump(inference, outfile)


if __name__ == '__main__':
    gen_answer(inference_file_name)
