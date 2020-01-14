import numpy as np

from functions import *
import functions as F

_fill = (128, 128, 128)

## TODO: use F

class RandomApply(object):

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            img = self.trans_func(img)
        return img


class AutoContrast(RandomApply):

    def __init__(self, p, cutoff=0):
        super(AutoContrast, self).__init__(p)
        self.cutoff = cutoff

    def trans_func(self, img):
        return F.autocontrast_func(img, self.cutoff)


class Equalize(RandomApply):

    def __init__(self, p):
        super(Equalize, self).__init__(p)

    def trans_func(self, img):
        return equalize_func(img)


class Invert(RandomApply):

    def __init__(self, p):
        super(Invert, self).__init__(p)

    def trans_func(self, img):
        return invert_func(img)


class Rotate(RandomApply):

    def __init__(self, p, degree):
        super(Rotate, self).__init__(p)
        self.degree = degree

    def trans_func(self, img):
        return rotate_func(img, self.degree, fill=_fill)


class Posterize(RandomApply):

    def __init__(self, p, bits):
        super(Posterize, self).__init__(p)
        self.bits = bits

    def trans_func(self, img):
        return posterize_func(img, self.bits)


class Solarize(RandomApply):

    def __init__(self, p, thresh=128):
        super(Solarize, self).__init__(p)
        self.thresh = thresh

    def trans_func(self, img):
        return solarize_func(img, self.thresh)


class Color(RandomApply):

    def __init__(self, p, factor=1.9):
        super(Color, self).__init__(p)
        self.factor = factor

    def trans_func(self, img):
        return color_func(img, self.factor)


class Contrast(RandomApply):

    def __init__(self, p, factor=1.9):
        super(Contrast, self).__init__(p)
        self.factor = factor

    def trans_func(self, img):
        return contrast_func(img, self.factor)


class Brightness(RandomApply):

    def __init__(self, p, factor=1.9):
        super(Brightness, self).__init__(p)
        self.factor = factor

    def trans_func(self, img):
        return brightness_func(img, self.factor)


class Sharpness(RandomApply):

    def __init__(self, p, factor=1.9):
        super(Sharpness, self).__init__(p)
        self.factor = factor

    def trans_func(self, img):
        return sharpness_func(img, self.factor)


class ShearX(RandomApply):

    def __init__(self, p, factor=1.9):
        super(ShearX, self).__init__(p)
        self.factor = factor

    def trans_func(self, img):
        return shear_x_func(img, self.factor, fill=_fill)


class ShearY(RandomApply):

    def __init__(self, p, factor=1.9):
        super(ShearY, self).__init__(p)
        self.factor = factor

    def trans_func(self, img):
        return shear_y_func(img, self.factor, fill=_fill)


class TranslateX(RandomApply):

    def __init__(self, p, offset=20):
        super(TranslateX, self).__init__(p)
        self.offset = offset

    def trans_func(self, img):
        return translate_x_func(img, self.offset, fill=_fill)


class TranslateY(RandomApply):

    def __init__(self, p, offset=20):
        super(TranslateY, self).__init__(p)
        self.offset = offset

    def trans_func(self, img):
        return translate_y_func(img, self.offset, fill=_fill)


class Cutout(RandomApply):

    def __init__(self, p, pad_size=20):
        super(Cutout, self).__init__(p)
        self.pad_size = pad_size

    def trans_func(self, img):
        return cutout_func(img, self.pad_size, replace=_fill)


class SolarizeAdd(RandomApply):

    def __init__(self, p, addition=20, thresh=128):
        super(SolarizeAdd, self).__init__(p)
        self.addition = addition
        self.thresh = thresh

    def trans_func(self, img):
        return solarized_add_func(img, self.addition, self.thresh)



if __name__ == "__main__":
    import cv2
    pth = './pic.jpg'
    mod = AutoContrast(p=0.0, cutoff=20)
    img_org = cv2.imread(pth)
    img_trans = mod(img_org)
    print(np.sum(np.abs(img_org - img_trans)))
