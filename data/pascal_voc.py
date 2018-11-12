import os.path as osp
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')


class VOCAnnotationTransform(object):

    def __init__(self, keep_difficult=False):
        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, w, h):
        labels = []

        points = ['xmin', 'ymin', 'xmax', 'ymax']
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            bndbox = []
            for i, point in enumerate(points):
                cur_point = int(bbox.find(point).text) - 1
                # scale height or width
                cur_point = cur_point / w if i % 2 == 0 else cur_point / h
                bndbox.append(cur_point)

            bndbox.append(self.class_to_ind[name])
            labels += [bndbox]

        return labels


class PascalVOC(Dataset):

    def __init__(self,
                 data_path,
                 image_sets,
                 new_size,
                 mode,
                 image_transform,
                 target_transform=VOCAnnotationTransform(),
                 keep_difficult=False):
        """
        Initialize dataset
        """

        self.data_path = data_path
        self.image_sets = image_sets
        self.new_size = new_size
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        self.annotation_path = osp.join('%s',
                                        self.mode,
                                        'Annotations',
                                        '%s.xml')
        self.image_path = osp.join('%s',
                                   self.mode,
                                   'JPegImages',
                                   '%s.jpg')
        self.text_path = osp.join('%s',
                                  self.mode,
                                  'ImageSets',
                                  'Main',
                                  '%s.txt')

        self.ids = []
        for (year, name) in self.image_sets:
            path = osp.join(self.data_path, 'VOC%s' % year)
            with open(self.text_path % (path, name)) as f:
                for line in f:
                    self.ids.append((path, line.strip()))

    def __len__(self):
        """
        returns the number of data in the dataset
        """
        return len(self.ids)

    def __getitem__(self, index):
        """
        return an item from the dataset
        """
        image, target, _, _ = self.pull_item(index)

        return image, target

    def pull_item(self, index):
        image_id = self.ids[index]

        target = ET.parse(self.annotation_path % image_id).getroot()
        image = cv2.imread(self.image_path % image_id)
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
        return cv2.imread(self.image_path % image_id, cv2.IMREAD_COLOR)

    def pull_annotation(self, index):
        image_id = self.ids[index]
        annotation = ET.parse(self.annotation_path % image_id).getroot()
        target = self.target_transform(annotation, 1, 1)
        return image_id[1], target

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
