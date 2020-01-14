import random

import numpy as np
import cv2

from .functions import *

max_level = 10
translate_const = 250
translate_bbox_const = 120
replace_value = (128, 128, 128)
cutout_const = 100
cutout_bbox_const = 50
cutout_max_pad_fraction = 0.75 # only used by cutout_bbox



##
## level to args
def translate_level_to_arg(translate_const):
    def level_to_arg(level):
        level = (level / max_level) * float(translate_const)
        if np.random.random() < 0.5: level = -level
        return [level, replace_value]
    return level_to_arg


def cutout_level_to_arg(cutout_const):
    def level_to_arg(level):
        level = int((level / max_level) * cutout_const)
        return [level, replace_value]
    return level_to_arg


def enhance_level_to_arg(level):
    return [0.1 + 1.8 * (level / max_level), ]


def shear_level_to_arg(level):
    level = 0.3 * (level / max_level)
    if np.random.random() < 0.5: level = -level
    return [level, replace_value]


def rotate_level_to_arg(level):
    level = 30 * (level / max_level)
    if np.random.random() < 0.5: level = -level
    return [level, replace_value]


def solarized_add_level_to_arg(level):
    level = int(110 * (level / max_level))
    return [level, ]


def solarize_level_to_arg(level):
    level = int(256 * (level / max_level))
    return [level, ]


def bbox_cutout_level_to_arg(cutout_max_pad_fraction):
    def func(level):
        return [(level / max_level) * cutout_max_pad_fraction, replace_value]
    return func


def posterize_level_to_arg(level):
    level = int(4 * (level / max_level))
    return [level, ]



##
## func and arg dicts
func_dict = {
    'TranslateX_BBox': translate_x_bbox_func,
    'TranslateY_BBox': translate_y_bbox_func,
    'Equalize': equalize_func,
    'Cutout': cutout_func,
    'Sharpness': sharpness_func,
    'ShearX_BBox': shear_x_bbox_func,
    'ShearY_BBox': shear_y_bbox_func,
    'TranslateY_Only_BBoxes': translate_y_only_bbox_func,
    'Rotate_BBox': rotate_bbox_func,
    'Color': color_func,

    ## not tried yet
    'ShearX_Only_BBoxes': shear_x_only_bbox_func,
    'ShearY_Only_BBoxes': shear_y_only_bbox_func,
    'Flip_Only_BBoxes': flip_only_bbox_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Cutout_Only_BBoxes': cutout_only_bbox_func,
    'SolarizeAdd': solarized_add_func,
    'Equalize_Only_BBoxes': equalize_only_bboxes_func,
    'AutoContrast': autocontrast_func,
    'Solarize': solarize_func,
    'BBox_Cutout': bbox_cutout_func,
    'Posterize': posterize_func,
}


arg_dict = {
    'TranslateX_BBox': translate_level_to_arg(translate_const=translate_const),
    'TranslateY_BBox': translate_level_to_arg(translate_const=translate_const),
    'Equalize': lambda level: (),
    'Cutout': cutout_level_to_arg(cutout_const=cutout_const),
    'Sharpness': enhance_level_to_arg,
    'ShearX_BBox': shear_level_to_arg,
    'ShearY_BBox': shear_level_to_arg,
    'TranslateY_Only_BBoxes': translate_level_to_arg(translate_const=translate_bbox_const),
    'Rotate_BBox': rotate_level_to_arg,
    'Color': enhance_level_to_arg,

    'ShearX_Only_BBoxes': shear_level_to_arg,
    'ShearY_Only_BBoxes': shear_level_to_arg,
    'Flip_Only_BBoxes': lambda level: (),
    'Contrast': enhance_level_to_arg,
    'Brightness': enhance_level_to_arg,
    'Cutout_Only_BBoxes': cutout_level_to_arg(cutout_const=cutout_bbox_const),
    'SolarizeAdd': solarized_add_level_to_arg,
    'Equalize_Only_BBoxes': lambda level: (),
    'AutoContrast': lambda level: (),
    'Solarize': solarize_level_to_arg,
    'BBox_Cutout': bbox_cutout_level_to_arg(cutout_max_pad_fraction),
    'Posterize': posterize_level_to_arg,
}


##
## searched policies
class ApplyPolicy(object):

    def __call__(self, img, bboxes):
        bboxes = self.get_valid_bboxes(img, bboxes)
        ops = random.choice(self.policies)
        for op in ops:
            name, prob, level = op
            if 'Only_BBoxes' in name:
                args = arg_dict[name](level)
                img, bboxes = func_dict[name](img, bboxes, prob, *args)
            elif np.random.random() < prob:
                args = arg_dict[name](level)
                img, bboxes = func_dict[name](img, bboxes, *args)
        return img, bboxes

    def get_valid_bboxes(self, img, bboxes):
        n_boxes = bboxes.shape[0]
        H, W, _ = img.shape
        bboxes[:, ::2] = bboxes[:, ::2].clip(0, W)
        bboxes[:, 1::2] = bboxes[:, 1::2].clip(0, H)
        ws = bboxes[:, 2] - bboxes[:, 0]
        hs = bboxes[:, 3] - bboxes[:, 1]
        ws, hs = ws.astype(np.int32), hs.astype(np.int32)
        valid_idx = np.logical_and(ws>=1, hs>=1)
        bboxes = bboxes[valid_idx]
        #
        #  for box in gt_bboxes:
        #      x1, y1, x2, y2 = box.tolist()
        #      w, h = x2-x1, y2-y1
        #      if w <= 0 or h <= 0:
        #          print('abnormal bbox: {}'.format(box))

        return bboxes


class PolicyV0(ApplyPolicy):

    def __init__(self):
        self.policies = [
            [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
            [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
            [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
            [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
            [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
        ]


class PolicyV1(ApplyPolicy):

    def __init__(self):
        self.policies = [
            [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
            [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
            [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
            [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
            [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
            [('Color', 0.0, 0), ('ShearX_Only_BBoxes', 0.8, 4)],
            [('ShearY_Only_BBoxes', 0.8, 2), ('Flip_Only_BBoxes', 0.0, 10)],
            [('Equalize', 0.6, 10), ('TranslateX_BBox', 0.2, 2)],
            [('Color', 1.0, 10), ('TranslateY_Only_BBoxes', 0.4, 6)],
            [('Rotate_BBox', 0.8, 10), ('Contrast', 0.0, 10)],
            [('Cutout', 0.2, 2), ('Brightness', 0.8, 10)],
            [('Color', 1.0, 6), ('Equalize', 1.0, 2)],
            [('Cutout_Only_BBoxes', 0.4, 6), ('TranslateY_Only_BBoxes', 0.8, 2)],
            [('Color', 0.2, 8), ('Rotate_BBox', 0.8, 10)],
            [('Sharpness', 0.4, 4), ('TranslateY_Only_BBoxes', 0.0, 4)],
            [('Sharpness', 1.0, 4), ('SolarizeAdd', 0.4, 4)],
            [('Rotate_BBox', 1.0, 8), ('Sharpness', 0.2, 8)],
            [('ShearY_BBox', 0.6, 10), ('Equalize_Only_BBoxes', 0.6, 8)],
            [('ShearX_BBox', 0.2, 6), ('TranslateY_Only_BBoxes', 0.2, 10)],
            [('SolarizeAdd', 0.6, 8), ('Brightness', 0.8, 10)],
        ]


class PolicyV2(ApplyPolicy):

    def __init__(self):
        self.policies = [
            [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)],
            [('Rotate_BBox', 0.4, 8), ('Sharpness', 0.4, 2),
            ('Rotate_BBox', 0.8, 10)],
            [('TranslateY_BBox', 1.0, 8), ('AutoContrast', 0.8, 2)],
            [('AutoContrast', 0.4, 6), ('ShearX_BBox', 0.8, 8),
            ('Brightness', 0.0, 10)],
            [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10),
            ('AutoContrast', 0.6, 0)],
            [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)],
            [('TranslateY_BBox', 0.0, 4), ('Equalize', 0.6, 8),
            ('Solarize', 0.0, 10)],
            [('TranslateY_BBox', 0.2, 2), ('ShearY_BBox', 0.8, 8),
            ('Rotate_BBox', 0.8, 8)],
            [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)],
            [('Color', 0.8, 4), ('TranslateY_BBox', 1.0, 6), ('Rotate_BBox', 0.6, 6)],
            [('Rotate_BBox', 0.6, 10), ('BBox_Cutout', 1.0, 4), ('Cutout', 0.2, 8)],
            [('Rotate_BBox', 0.0, 0), ('Equalize', 0.6, 6), ('ShearY_BBox', 0.6, 8)],
            [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2),
            ('Brightness', 0.2, 2)],
            [('TranslateY_BBox', 0.4, 8), ('Solarize', 0.4, 6),
            ('SolarizeAdd', 0.2, 10)],
            [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
        ]


class PolicyV3(ApplyPolicy):

    def __init__(self):
        self.policies = [
            [('Posterize', 0.8, 2), ('TranslateX_BBox', 1.0, 8)],
            [('BBox_Cutout', 0.2, 10), ('Sharpness', 1.0, 8)],
            [('Rotate_BBox', 0.6, 8), ('Rotate_BBox', 0.8, 10)],
            [('Equalize', 0.8, 10), ('AutoContrast', 0.2, 10)],
            [('SolarizeAdd', 0.2, 2), ('TranslateY_BBox', 0.2, 8)],
            [('Sharpness', 0.0, 2), ('Color', 0.4, 8)],
            [('Equalize', 1.0, 8), ('TranslateY_BBox', 1.0, 8)],
            [('Posterize', 0.6, 2), ('Rotate_BBox', 0.0, 10)],
            [('AutoContrast', 0.6, 0), ('Rotate_BBox', 1.0, 6)],
            [('Equalize', 0.0, 4), ('Cutout', 0.8, 10)],
            [('Brightness', 1.0, 2), ('TranslateY_BBox', 1.0, 6)],
            [('Contrast', 0.0, 2), ('ShearY_BBox', 0.8, 0)],
            [('AutoContrast', 0.8, 10), ('Contrast', 0.2, 10)],
            [('Rotate_BBox', 1.0, 10), ('Cutout', 1.0, 10)],
            [('SolarizeAdd', 0.8, 6), ('Equalize', 0.8, 8)],
        ]



if __name__ == '__main__':
    im = cv2.imread('./leon.jpg')
    poly = SubPolicy1()
    im, _ = poly(im, np.empty((10, 4)))

    import json
    jpth = '/home/coin/Documents/Work/Haishen/log/detectron/lib/datasets/data/coco/annotations/instances_minival2014.json'
    imgpth = '/home/coin/Documents/Work/Haishen/detectron/detectron/lib/datasets/data/coco/coco_val2014/'
    with open(jpth, 'r') as fr:
        jobj = json.load(fr)
        img = jobj['images'][2]
        imid = img['id']
        anns = [ann for ann in jobj['annotations'] if ann['image_id'] == imid]
    impath = imgpth + img['file_name']
    im = cv2.imread(impath)
    gt_bboxes = []
    for ann in anns:
        x1, y1, w, h = [int(el) for el in ann['bbox']]
        x2, y2 = x1 + w, y1 + h
        gt_bboxes.append([x1, y1, x2, y2])
        #  print(x1, y1, x2, y2)
        #  cv2.rectangle(im, (x1, y1), (x2, y2), (155, 0, 0), 3)
    gt_bboxes = np.array(gt_bboxes)

    #  poly = SubPolicy3()
    #  poly.p1 = 1
    #  poly.m1 = 20
    #  poly.p2 = 1
    #  #  poly.m2 = 0.5
    #  im, gt_bboxes = poly(im, gt_bboxes)
    im, gt_bboxes = do_learned_augs(im, gt_bboxes)
    for box in gt_bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(im, (x1, y1), (x2, y2), (155, 0, 0), 3)

    cv2.imshow('org', im)
    cv2.waitKey(0)


