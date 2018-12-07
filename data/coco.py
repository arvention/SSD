import json
import pickle

import cv2
import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Coco(Dataset):

    def __init__(self,
                 data_path,
                 image_sets,
                 image_transform,
                 target_transform=None):
        self.data_path = data_path
        self.image_sets = image_sets
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.view_map = {
            'minical2014': 'val2014',
            'valminusminival2014': 'val2014',
            'test-dev2015': 'test2015'
        }

        for (year, image_set) in image_sets:
            coco_name = image_set + year
            data_name = coco_name
            if coco_name in self.view_map:
                data_name = self.view_map[coco_name]

            annotation_file = self.get_annotation_file(coco_name)
            self.dataset = COCO(annotation_file)
            self.coco_name = coco_name

            classes = self.dataset.loadCats(self.dataset.getCatIds())
            self.classes = tuple(['__bg__'] + [c['name'] for c in classes])
            self.class_count = len(self.classes)
            self.class_to_index = dict(zip(self.classes,
                                           range(self.class_count)))
            self.class_to_id = dict(zip([c['name'] for c in classes],
                                        self.dataset.getCatIds()))

            self.indices = self.dataset.getImgIds()
            self.ids = []
            self.annotations = []
            ids = [self.get_path(data_name, index) for index in self.indices]
            self.ids.extend(ids)

            if image_set.find('test') != -1:
                print('test set will not load annotations')
            else:
                annotations = self.load_annotations(coco_name,
                                                    self.indices,
                                                    self.dataset)
                self.annotations.extend(annotations)

    def get_path(self, name, index):
        if '2014' in name or '2015' in name:
            file_name = ('COCO_' + name + '_' + str(index).zfill(12) + '.jpg')
            image_path = osp.join(self.data_path, name, file_name)

        if '2017' in name:
            file_name = str(index).zfill(12) + '.jpg'
            image_path = osp.join(self.data_path, name, file_name)

        return image_path

    def get_annotation_file(self, name):
        prefix = None
        if name.find('test') == -1:
            prefix = 'instances'
        else:
            prefix = 'image_info'

        annotation_file = osp.join(self.data_path,
                                   'annotations',
                                   prefix + '_' + name + '.json')

        return annotation_file

    def load_annotations(self, coco_name, indices, dataset):
        cache_file = osp.join(self.data_path, coco_name + '_gt_roidb.pkl')
        roidb = None

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = pickle.load(f)
            print('{} gt roidb loaded from {}'.format(coco_name, cache_file))

        else:
            roidb = [self.get_annotation_from_index(index, dataset)
                     for index in indices]
            with open(cache_file, 'wb') as f:
                pickle.dump(roidb, f, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))

        return roidb

    def get_annotation_from_index(self, index, dataset):
        annotation = dataset.loadImgs(index)[0]
        width = annotation['width']
        height = annotation['height']

        annotation_ids = dataset.getAnnIds(imgIds=index, iscrowd=None)
        objects = dataset.loadAnns(annotation_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objects:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        id_to_index = dict([(self.class_to_id[c],
                             self.class_to_index[c])
                            for c in self.classes[1:]])

        for ix, obj in enumerate(objs):
            c = id_to_index[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = c

        return res

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        target = self.annotations[index]

        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.image_transform is not None:
            image, target = self.preproc(image, target)

        return image, target

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(img_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
              '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__bg__':
                continue
            # minus 1 because of __bg__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_indexes):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__bg__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             self.num_classes))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
            '''
            if cls_ind ==30:
                res_f = res_file+ '_1.json'
                print('Writing results json to {}'.format(res_f))
                with open(res_f, 'w') as fid:
                    json.dump(results, fid)
                results = []
            '''
        # res_f2 = res_file+'_2.json'
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self.coco_name +
                                             '_results'))
        res_file += '.json'
        self.write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            self.do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
