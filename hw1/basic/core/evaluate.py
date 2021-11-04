from matplotlib import pyplot as plt
from config import *
from core.datasets import *
import numpy as np
import pandas as pd


def predict_dataset(model, dataloader):
    model.eval()
    pred_list = list()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            pred_list.extend(preds.tolist())
    return pred_list


def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(dataloader.dataset)
    return accuracy


def generate_hw_submission(model, val_transform):
    unlabeled_dataset = BirdDataset(
        annotations_file=os.path.join(DATA_PATH_, 'false_answer.txt'),
        img_dir=os.path.join(DATA_PATH_, 'testing_images/'),
        transform=val_transform,
        target_transform=None)

    unlabeled_dataloader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_images = unlabeled_dataset.img_labels[0].values.tolist()
    class_names = pd.read_csv(
        os.path.join(
            DATA_PATH_,
            'training_labels.txt'),
        sep=' ',
        header=None)[1].sort_values().unique()
    myanswer_index = predict_dataset(model, unlabeled_dataloader)
    myanswer_class_names = class_names[myanswer_index]

    submission = []
    for image_name, image_pred in zip(test_images, myanswer_class_names):
        submission.append([image_name, image_pred])
    np.savetxt('answer.txt', submission, fmt='%s')


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)
