# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


import time
import shutil
import numpy as np
import cv2

import selectivesearch
from utils.util import check_dir
from utils.util import parse_car_csv
from utils.util import parse_xml
from utils.util import compute_ious


# train
# positive num: 66517
# negatie num: 464340
# val
# positive num: 64712
# negative num: 415134


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):

    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy='q')

    rects = selectivesearch.get_rects(gs)

    bndboxs = parse_xml(annotation_path)


    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size


    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if iou_list[i] >= 0.5:
            positive_list.append(rects[i])
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list


if __name__ == '__main__':
    car_root_dir = '../../data/voc_car/'
    finetune_root_dir = '../../data/finetune_car/'
    check_dir(finetune_root_dir)

    gs = selectivesearch.get_selective_search()
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)

        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')

            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')

            shutil.copyfile(src_jpeg_path, dst_jpeg_path)

            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')
