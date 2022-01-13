import os
import cv2
import random

src_path = 'training_hr_images/training_hr_images'
tar_tarin_path = 'KAIR/trainsets/trainH'
tar_test_path = 'KAIR/testsets/testH'
if not os.path.exists(tar_test_path):
    os.mkdir(tar_test_path)


test_size = 15
random.seed(2022)

images_list = os.listdir(src_path)
random.shuffle(images_list)
train_image_list = images_list[:-test_size]
test_image_list = images_list[-test_size:]

for image in train_image_list:
    dat = cv2.imread(os.path.join(src_path,image))
    cv2.imwrite(os.path.join(tar_tarin_path,image),dat)

for image in test_image_list:
    dat = cv2.imread(os.path.join(src_path,image))
    cv2.imwrite(os.path.join(tar_test_path,image),dat)