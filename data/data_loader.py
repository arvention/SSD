import torch
from torch.utils.data import DataLoader
from data.pascal_voc import PascalVOC
from data.coco import Coco
from data.coco import CocoAnnotationTransform as coco_annotation
from data.augmentations import Augmentations, BaseTransform


VOC_CONFIG = {
    '0712': ([('2007', 'trainval'), ('2012', 'trainval')],
             [('2007', 'test')]),
    '0712+': ([('2007', 'trainval'), ('2012', 'trainval'), ('2007', 'test')],
              [('2012', 'test')])
}

COCO_CONFIG = {
    '2014': ('train2014',
             'val2014'),
    '2017': ('train2017',
             'val2017')
}


def detection_collate(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images, 0), targets


def get_loader(config):
    """
    returns train and test data loader
    """

    train_dataset = None
    test_dataset = None

    train_loader = None
    test_loader = None

    batch_size = config.batch_size
    new_size = config.new_size
    means = config.means

    if config.dataset == 'voc':
        voc_config = VOC_CONFIG[config.voc_config]
        if config.mode == 'train' or config.mode == 'trainval':
            image_transform = Augmentations(new_size, means)
            train_dataset = PascalVOC(data_path=config.voc_data_path,
                                      image_sets=voc_config[0],
                                      new_size=new_size,
                                      mode='trainval',
                                      image_transform=image_transform)

        elif config.mode == 'test':
            image_transform = BaseTransform(new_size, means)
            test_dataset = PascalVOC(data_path=config.voc_data_path,
                                     image_sets=voc_config[1],
                                     new_size=new_size,
                                     mode=config.mode,
                                     image_transform=image_transform)

    elif config.dataset == 'coco':
        coco_config = COCO_CONFIG[config.coco_config]
        target_transform = coco_annotation(data_path=config.coco_data_path)
        if config.mode == 'train':
            image_transform = Augmentations(new_size, means)
            train_dataset = Coco(data_path=config.coco_data_path,
                                 image_set=coco_config[0],
                                 image_transform=image_transform,
                                 target_transform=target_transform)

        elif config.mode == 'test':
            image_transform = BaseTransform(new_size, means)
            test_dataset = Coco(data_path=config.coco_data_path,
                                image_set=coco_config[1],
                                image_transform=image_transform,
                                target_transform=target_transform)

    if train_dataset is not None:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  num_workers=4,
                                  pin_memory=True)

    if test_dataset is not None:
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=detection_collate,
                                 num_workers=4,
                                 pin_memory=True)

    return train_loader, test_loader
