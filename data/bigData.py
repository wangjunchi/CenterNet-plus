'''
Dataset for The 3rd IKCEST BigData Competition

author: junchi
date: 2021/8/2
'''


import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import cv2

from pycocotools.coco import COCO

from data import create_gt


class_labels = ("Motor Vehicle", "Non-motorized Vehicle", "Pedestrian", "Traffic Light-Red Light",
                "Traffic Light-Yellow Light", "Traffic Light-Green Light", "Traffic Light-Off")

class_index = [1, 2, 3, 4, 5, 6, 7]


class BigDataDataset(Dataset):
    '''
    Dataset class
    '''
    def __init__(self,
                 root_dir='/media/junchi/Data/BigData2021/dataset/train_COCO/detection',
                 img_dir="data",
                 json_file="train.json",
                 img_size=(1280, 720),
                 train=True,
                 stride=32,
                 transform=None,
                 base_transform=None,
                 mosaic=False):
        '''
        Args:
            root_dir: the root dir for the dataset
            img_dir: the dir for all images
            json_file: the file of all annotations
            train: training dataset or test dataset
            img_size: resize all images to the same size
            stride:
            transform: simple transform as data augmentation
            mosaic: use mosaic for data augmentation
        '''
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.json_file = json_file
        self.train = train
        self.img_size = img_size
        self.stride = stride
        self.transform = transform
        self.base_transform = base_transform
        self.mosaic = mosaic

        #self.coco = COCO(self.root_dir+self.json_file)
        self.coco = COCO(os.path.join(self.root_dir, self.json_file))
        self.image_ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())

    def __len__(self):
        return len(self.image_ids)

    def pull_image(self, index):
        id_ = self.image_ids[index]
        image_file = os.path.join(self.root_dir, self.img_dir, '{:04d}'.format(id_)+'.jpg')
        img = cv2.imread(image_file)

        return img, id

    def pull_anno(self, index):
        id_ = self.image_ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        target = []
        for anno in annotations:
            if 'bbox' in anno:
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = xmin + anno['bbox'][2]
                ymax = ymin + anno['bbox'][3]

                if anno['area'] > 0 and xmax >= xmin and ymax >= ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    target.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!')

        return target

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        if self.train:
            gt_tensor = create_gt.gt_creator(img_size=self.img_size,
                                             stride=self.stride,
                                             num_classes=80,
                                             label_lists=gt)
            return im, gt_tensor
        else:
            return im, gt

    def pull_item(self, index):
        id_ = self.image_ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.root_dir, self.img_dir, '{:05d}'.format(id_)+'.jpg')
        print("image file = " + img_file)
        img = cv2.imread(img_file)

        assert img is not None

        height, width, channels = img.shape

        # COCOAnnotation Transform
        # start here :
        target = []
        for anno in annotations:
            x1 = np.max((0, anno['bbox'][0]))
            y1 = np.max((0, anno['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
            if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                label_ind = anno['category_id']
                cls_id = self.class_ids.index(label_ind)
                x1 /= width
                y1 /= height
                x2 /= width
                y2 /= height

                target.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
        # end here .
        # mosaic augmentation
        if self.mosaic and np.random.randint(2):
            ids_list_ = self.image_ids[:index] + self.image_ids[index + 1:]
            # random sample 3 indexs
            id2, id3, id4 = random.sample(ids_list_, 3)
            ids = [id2, id3, id4]
            img_lists = [img]
            tg_lists = [target]
            # load other 3 images and targets
            for id_ in ids:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
                annotations = self.coco.loadAnns(anno_ids)

                # load image and preprocess
                img_file = os.path.join(self.root_dir, self.img_dir, '{:04d}'.format(id_)+'.jpg')
                img_i = cv2.imread(img_file)

                assert img_i is not None

                height_i, width_i, channels_i = img_i.shape
                # COCOAnnotation Transform
                # start here :
                target_i = []
                for anno in annotations:
                    x1 = np.max((0, anno['bbox'][0]))
                    y1 = np.max((0, anno['bbox'][1]))
                    x2 = np.min((width_i - 1, x1 + np.max((0, anno['bbox'][2] - 1))))
                    y2 = np.min((height_i - 1, y1 + np.max((0, anno['bbox'][3] - 1))))
                    if anno['area'] > 0 and x2 >= x1 and y2 >= y1:
                        label_ind = anno['category_id']
                        cls_id = self.class_ids.index(label_ind)
                        x1 /= width_i
                        y1 /= height_i
                        x2 /= width_i
                        y2 /= height_i

                        target_i.append([x1, y1, x2, y2, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
                # end here .
                img_lists.append(img_i)
                tg_lists.append(target_i)

            mosaic_img = np.zeros([self.img_size * 2, self.img_size * 2, img.shape[2]], dtype=np.uint8)
            # mosaic center
            yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in
                      [-self.img_size // 2, -self.img_size // 2]]
            # yc = xc = self.img_size

            mosaic_tg = []
            for i in range(4):
                img_i, target_i = img_lists[i], tg_lists[i]
                h0, w0, _ = img_i.shape

                # resize image to img_size
                img_i = cv2.resize(img_i, (self.img_size, self.img_size))
                h, w, _ = img_i.shape

                # place img in img4
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                else:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

                mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b

                # labels
                target_i = np.array(target_i)
                target_i_ = target_i.copy()
                if len(target_i) > 0:
                    # a valid target, and modify it.
                    target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                    target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                    target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                    target_i_[:, 3] = (h * (target_i[:, 3]) + padh)

                    mosaic_tg.append(target_i_)

            if len(mosaic_tg) == 0:
                mosaic_tg = np.zeros([1, 5])
            else:
                mosaic_tg = np.concatenate(mosaic_tg, axis=0)
                # Cutout/Clip targets
                np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
                # normalize
                mosaic_tg[:, :4] /= (self.img_size * 2)

            # augment
            mosaic_img, boxes, labels = self.base_transform(mosaic_img, mosaic_tg[:, :4], mosaic_tg[:, 4])
            # to rgb
            mosaic_img = mosaic_img[:, :, (2, 1, 0)]
            mosaic_tg = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            return torch.from_numpy(mosaic_img).permute(2, 0, 1).float(), mosaic_tg, self.img_size, self.img_size

        # basic augmentation(SSDAugmentation or BaseTransform)
        else:
            # check targets
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            return torch.from_numpy(img).permute(2, 0, 1), target, height, width

if __name__ == "__main__":
    def base_transform(image, size, mean):
        x = cv2.resize(image, size).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels


    img_size = (1280, 720)
    dataset = BigDataDataset(
        img_size=img_size,
        transform=BaseTransform(img_size, (0, 0, 0)),
        base_transform=BaseTransform(img_size, (0,0,0))
    )

    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')

        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size[0]
            ymin *= img_size[1]
            xmax *= img_size[0]
            ymax *= img_size[1]
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)