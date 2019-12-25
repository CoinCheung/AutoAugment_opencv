

import numpy as np
from .functions import func_dict
from .parse_aug_args import ParseAugArgs



class RandomAugment(object):

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.args_parser = ParseAugArgs(translate_const=100, cutout_const=40)

    def get_random_ops(self):
        sampled_ops = np.random.choice(list(func_dict.keys()), self.N)
        return [(op, self.M) for op in sampled_ops]

    def __call__(self, img):
        ops = self.get_random_ops()
        for name, level in ops:
            args = self.args_parser(name, level)
            img = func_dict[name](img, *args)
        return img


if __name__ == '__main__':
    import cv2
    pth = './pic.jpg'
    imcv = cv2.imread(pth)

    RA = RandomAugment(N=2, M=9)
    for i in range(1000):
        RA(imcv)
