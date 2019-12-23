

import numpy as np



def enhance_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1, )
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)
    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)
    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)
        return (level, replace_value)
    return level_to_args


def none_level_to_args(level):
    return ()


def rotate_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30
        if np.random.random() < 0.5: level = -level
        return (level, replace_value)
    return level_to_args


def posterize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)
        return (level, )
    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)
        return (level, )
    return level_to_args


def solarize_add_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 110)
        return (level, 128)
    return level_to_args


class ParseAugArgs(object):

    def __init__(self, translate_const, cutout_const, MAX_LEVEL=10, replace_value=(128, 128, 128)):
        self.arg_dict = {
            'AutoContrast': none_level_to_args,
            'Equalize': none_level_to_args,
            'Invert': none_level_to_args,
            'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
            'Posterize': posterize_level_to_args(MAX_LEVEL),
            'Solarize': solarize_level_to_args(MAX_LEVEL),
            'Color': enhance_level_to_args(MAX_LEVEL),
            'Contrast': enhance_level_to_args(MAX_LEVEL),
            'Brightness': enhance_level_to_args(MAX_LEVEL),
            'Sharpness': enhance_level_to_args(MAX_LEVEL),
            'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
            'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
            'TranslateX': translate_level_to_args(
                translate_const, MAX_LEVEL, replace_value
            ),
            'TranslateY': translate_level_to_args(
                translate_const, MAX_LEVEL, replace_value
            ),
            'Cutout': cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value),
            'SolarizeAdd': solarize_add_level_to_args(MAX_LEVEL),
        }

    def __call__(self, name, level):
        return self.arg_dict[name](level)


if __name__ == '__main__':
    import cv2
    pth = './pic.jpg'
    imcv = cv2.imread(pth)

    MAX_LEVEL = 10
    replace_value = (128, 128, 128)
    translate_const = 100
    cutout_const = 40

    N, M = 2, 9
    arg_dict = ParseArgs(translate_const, cutout_const)
    RA = RandomAugment(N, M, arg_dict=arg_dict)
    for i in range(1000):
        RA(imcv)
