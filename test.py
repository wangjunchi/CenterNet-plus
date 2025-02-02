import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import numpy as np
import cv2
import time


parser = argparse.ArgumentParser(description='CenterNet-plus Detection')
parser.add_argument('-v', '--version', default='centernet_plus',
                    help='centernet_plus')
parser.add_argument('-bk', '--backbone', default='r18',
                    help='r18, r34, r50, r101')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=512, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-vs', '--visual_threshold', default=0.25, type=float,
                    help='Final confidence threshold')
parser.add_argument('--conf_thresh', default=0.25, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')

args = parser.parse_args()


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s' % (class_names[int(cls_indx)])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
        

def test(net, device, testset, transform, thresh, class_colors=None, class_names=None, class_indexs=None, dataset='voc'):
    num_images = len(testset)
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset)
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)
        print('Saving the' + str(index) + '-th image ...')
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img)


if __name__ == '__main__':
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = args.input_size

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(root=VOC_ROOT, 
                               image_sets=[('2007', 'test')], 
                               transform=None)

    elif args.dataset == 'coco-val':
        print('test on coco-val ...')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
                    data_dir=coco_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=input_size)

    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(num_classes)]

    # load net
    if args.version == 'centernet_plus':
        from models.centernet_plus import CenterNetPlus
        net = CenterNetPlus(device=device, 
                          input_size=input_size, 
                          num_classes=num_classes,
                          backbone=args.backbone,
                          conf_thresh=args.conf_thresh, 
                          nms_thresh=args.nms_thresh, 
                          use_nms=args.use_nms
                          )

    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=net, 
        device=device, 
        testset=dataset,
        transform=BaseTransform(input_size),
        thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        dataset=args.dataset
        )
