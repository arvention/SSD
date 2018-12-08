import os.path as osp
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pycocotools.coco import COCO


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    with open(label_file, 'r') as f:
        for line in f:
            ids = line.split(',')
            label_map[int(ids[0])] = int(ids[1])
    return label_map


class CocoAnnotationTransform(object):
    def __init__(self, data_path):
        self.label_map = get_label_map(osp.join(data_path, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        labels = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                labels += [final_box]
            else:
                print("no bbox problem!")

        return labels


class Coco(Dataset):

    def __init__(self,
                 data_path,
                 image_set,
                 image_transform,
                 target_transform):

        self.data_path = data_path
        self.image_set = image_set
        self.dataset = COCO(osp.join(data_path,
                                     'Annotations',
                                     'instances_{}.json'.format(image_set)))
        self.ids = [self.dataset.imgToAnns.keys()]
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        return an item from the dataset
        """
        image, target, _, _ = self.pull_item(index)

        return image, target

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        image_id = self.ids[index]
        target = self.dataset.imgToAnns[image_id]
        annotation_ids = self.dataset.getAnnIds(imgIds=image_id)

        target = self.dataset.loadAnns(annotation_ids)
        path = osp.join(self.data_path,
                        self.image_set,
                        self.dataset.loadImgs(image_id)[0]['file_name'])

        image = cv2.imread(path)
        height, width, _ = image.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.image_transform is not None:
            target = np.array(target)
            boxes = target[:, :4]
            labels = target[:, 4]
            image, boxes, labels = self.image_transform(image, boxes, labels)
            # to rgb
            image = image[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        image_id = self.ids[index]
        path = osp.join(self.data_path,
                        self.image_set,
                        self.dataset.loadImgs(image_id)[0]['file_name'])
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def pull_annotation(self, index):
        image_id = self.ids[index]
        annotation_ids = self.dataset.getAnnIds(image_id)
        return self.dataset.loadAnns(annotation_ids)
