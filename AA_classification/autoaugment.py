

import numpy as np
import random
from .functions import func_dict
from .parse_aug_args import ParseAugArgs



class AutoAugment(object):

    def __init__(self):
        self.policies = [
            [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
            [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
            [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
            [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
            [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
            [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
            [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
            [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
            [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
            [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
            [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
            [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
            [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
            [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
            [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
            [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
            [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
            [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
            [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
            [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
            [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
            [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
            [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
            [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
            [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
        ]

        self.args_parser = ParseAugArgs(translate_const=250, cutout_const=100)

    def __call__(self, img):
        subpolicy = random.choice(self.policies)
        for name, prob, level in subpolicy:
            if np.random.random() < prob:
                args = self.args_parser(name, level)
                img = func_dict[name](img, *args)
        return img


if __name__ == '__main__':
    import cv2
    pth = './pic.jpg'
    imcv = cv2.imread(pth)

    AA = AutoAugment()
    for i in range(1000):
        AA(imcv)
