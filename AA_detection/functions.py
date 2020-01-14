
import cv2
import numpy as np



## functions of AA for detection

def warp_image_and_bbox(img, bboxes, M, fill=(0, 0, 0)):
    rows, cols, _ = img.shape
    n_boxes = bboxes.shape[0]
    # warp image
    img = cv2.warpAffine(img, M, (cols, rows), borderValue=fill)
    # warp bboxes
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)
    x2 = bboxes[:, 2].reshape(-1, 1)
    y2 = bboxes[:, 3].reshape(-1, 1)
    xys = np.hstack([
        x1, y1, x1, y2, x2, y2, x2, y1
    ]).reshape(-1, 2)
    xys = np.hstack([xys, np.ones((4*n_boxes, 1))]).T
    xys = np.matmul(M, xys).T
    xys = xys[:, :2].reshape(-1, 8)
    xs = xys[:, [0, 2, 4, 6]]
    ys = xys[:, [1, 3, 5, 7]]
    xmin = np.clip(np.min(xs, 1).reshape(-1, 1), 0, cols-1)
    xmax = np.clip(np.max(xs, 1).reshape(-1, 1), 0, cols-1)
    ymin = np.clip(np.min(ys, 1).reshape(-1, 1), 0, rows-1)
    ymax = np.clip(np.max(ys, 1).reshape(-1, 1), 0, rows-1)
    bboxes = np.hstack([xmin, ymin, xmax, ymax]).astype(np.float32)
    return img, bboxes


def warp_only_bbox(img, bbox, M, fill=(0, 0, 0)):
    x1, y1, x2, y2 = bbox.astype(np.int64).tolist()
    rows, cols = y2 - y1 + 1, x2 - x1 + 1
    img[y1:y2+1, x1:x2+1, :] = cv2.warpAffine(
        img[y1:y2+1, x1:x2+1, :], M, (cols, rows), borderValue=fill
    )
    return img


def apply_to_bboxes_only(img, bboxes, func, p):
    n_boxes = bboxes.shape[0]
    for i in range(n_boxes):
        if np.random.random() < p:
            img = func(img, bboxes[i])
    return img, bboxes


def get_bboxes_only_prob(p):
    return p / 3



def translate_x_bbox_func(img, bboxes, level, fill=(0, 0, 0)):
    #  rows, cols = img.shape[0], img.shape[1]
    #  M = np.float32([[1, 0, level], [0, 1, 0]])
    #  img = cv2.warpAffine(img, M, (cols, rows))
    #  bboxes[..., 0] += level
    #  bboxes[..., 2] += level
    #  bboxes[..., 0] = np.clip(bboxes[..., 0], 0, cols-1)
    #  bboxes[..., 2] = np.clip(bboxes[..., 2], 0, cols-1)
    #  return img, bboxes
    M = np.float32([[1, 0, level], [0, 1, 0]])
    return warp_image_and_bbox(img, bboxes, M, fill=fill)


def translate_y_bbox_func(img, bboxes, level, fill=(0, 0, 0)):
    M = np.float32([[1, 0, 0], [0, 1, level]])
    return warp_image_and_bbox(img, bboxes, M, fill=fill)


def equalize_func(img, bboxes):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256
    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0: return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out, bboxes


def cutout_func(img, bboxes, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out, bboxes


def sharpness_func(img, bboxes, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out, bboxes


def shear_x_bbox_func(img, bboxes, level, fill=(0, 0, 0)):
    M = np.float32([[1, level, 0], [0, 1, 0]])
    return warp_image_and_bbox(img, bboxes, M, fill=fill)


def shear_y_bbox_func(img, bboxes, level, fill=(0, 0, 0)):
    M = np.float32([[1, 0, 0], [level, 1, 0]])
    return warp_image_and_bbox(img, bboxes, M, fill=fill)


def translate_y_only_bbox_func(img, bboxes, p, level, fill=(0, 0, 0)):
    M = np.float32([[1, 0, 0], [0, 1, level]])
    p = get_bboxes_only_prob(p)
    func = lambda img, bbox: warp_only_bbox(img, bbox, M, fill=fill)
    #  def func(img, bbox):
    #      M = np.float32([[1, 0, 0], [0, 1, level]])
    #      img = warp_only_bbox(img, bbox, M, fill=fill)
    #      return img
    return apply_to_bboxes_only(img, bboxes, func, p)


def rotate_bbox_func(img, bboxes, level, fill=(0, 0, 0)):
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), level, 1.)
    return warp_image_and_bbox(img, bboxes, M, fill=fill)


def color_func(img, bboxes, factor):
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out, bboxes


def shear_x_only_bbox_func(img, bboxes, p, level, fill=(0, 0, 0)):
    M = np.float32([[1, level, 0], [0, 1, 0]])
    p = get_bboxes_only_prob(p)
    func = lambda img, bbox: warp_only_bbox(img, bbox, M, fill=fill)
    return apply_to_bboxes_only(img, bboxes, func, p)


def shear_y_only_bbox_func(img, bboxes, p, level, fill=(0, 0, 0)):
    M = np.float32([[1, 0, 0], [level, 1, 0]])
    p = get_bboxes_only_prob(p)
    func = lambda img, bbox: warp_only_bbox(img, bbox, M, fill=fill)
    return apply_to_bboxes_only(img, bboxes, func, p)


def flip_only_bbox_func(img, bboxes, p):
    p = get_bboxes_only_prob(p)
    def func(img, bbox):
        x1, y1, x2, y2 = bbox.tolist()
        img[y1:y2+1, x1:x2+1, :] = img[y1:y2+1, x1:x2+1, :][:, ::-1, :]
        return img
    return apply_to_bboxes_only(img, bboxes, func, p)


def contrast_func(img, bboxes, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = np.array([(
        el -74) * factor + 74
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out, bboxes


def brightness_func(img, bboxes, factor):
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out, bboxes


def cutout_within_bbox(img, bbox, pad_size, fill=(0, 0, 0)):
    x1, y1, x2, y2 = bbox.tolist()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    rh, rw = np.random.random(2)
    ch, cw = y1 + int(rh * h), x1 + int(rw * w)
    if isinstance(pad_size, (tuple, list)):
        padx, pady = pad_size
    else:
        padx = pady = pad_size
    bx1, bx2 = int(max(cw - padx, x1)), int(min(cw + padx, x2))
    by1, by2 = int(max(ch - pady, y1)), int(min(ch + pady, y2))
    img[by1:by2+1, bx1:bx2+1, :] = fill
    return img


def cutout_only_bbox_func(img, bboxes, p, pad_size, fill=(0, 0, 0)):
    p = get_bboxes_only_prob(p)
    def func(img, bbox):
        return cutout_within_bbox(img, bbox, pad_size, fill)
    return apply_to_bboxes_only(img, bboxes, func, p)


def solarized_add_func(img, bboxes, addition, thresh=128):
    table = np.array([
        el + addition if el < thresh else el for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out, bboxes


def equalize_only_bboxes_func(img, bboxes, p):
    p = get_bboxes_only_prob(p)
    n_bins = 256
    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0: return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]
    equalize = lambda img: cv2.merge([tune_channel(ch) for ch in cv2.split(img)])

    def func(img, bbox):
        x1, y1, x2, y2 = bbox.tolist()
        img[y1:y2+1, x1:x2+1, :] = equalize(img[y1:y2+1, x1:x2+1, :])
        return img

    return apply_to_bboxes_only(img, bboxes, func, p)


def autocontrast_func(img, bboxes, cutoff=0):
    n_bins = 256
    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out, bboxes


def solarize_func(img, bboxes, thresh=128):
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out, bboxes


def bbox_cutout_func(img, bboxes, cutout_pad_fraction, fill):
    n_bboxes = bboxes.shape[0]
    if n_bboxes > 0:
        idx = np.random.choice(n_bboxes)
        w = bboxes[idx][2] - bboxes[idx][0] + 1
        h = bboxes[idx][3] - bboxes[idx][1] + 1
        pad_size = (int(w * cutout_pad_fraction), int(h * cutout_pad_fraction))
        out = cutout_within_bbox(img, bboxes[idx], pad_size, fill)
    return out, bboxes


def posterize_func(img, bboxes, bits):
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out, bboxes



if __name__ == '__main__':
    import PIL
    from PIL import Image, ImageEnhance, ImageOps
    import time

    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = sharpness_func(imcv, 0.3).astype(np.float32)
    sharp_pil = ImageEnhance.Sharpness(impil)
    out_pil = np.array(sharp_pil.enhance(0.3))[:, :, ::-1].astype(np.float32)
    print('sharpness')
    print(np.sum(np.abs(out_pil - out_cv)))
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    t1 = time.time()
    n_test = 100
    for i in range(n_test):
        out_cv = autocontrast_func(imcv, 20).astype(np.float32)
    t2 = time.time()
    for i in range(n_test):
        out_pil = np.array(ImageOps.autocontrast(impil, cutoff=20))[:, :, ::-1]
    t3 = time.time()
    print('autocontrast')
    print('cv time: {}'.format(t2 - t1))
    print('pil time: {}'.format(t3 - t2))
    print(np.sum(np.abs(out_pil - out_cv)))
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = equalize_func(imcv).astype(np.float32)
    out_pil = np.array(ImageOps.equalize(impil))[:, :, ::-1]
    print('equalize')
    print(np.sum(np.abs(out_pil - out_cv)))
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = invert_func(imcv).astype(np.float32)
    out_pil = np.array(ImageOps.invert(impil))[:, :, ::-1]
    print('invert')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ###
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = rotate_func(imcv, 10).astype(np.float32)
    out_pil = np.array(impil.rotate(10))[:, :, ::-1].astype(np.float32)
    print('rotate')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = posterize_func(imcv, 3).astype(np.float32)
    out_pil = np.array(ImageOps.posterize(impil, 3))[:, :, ::-1]
    print('posterize')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = solarize_func(imcv, 70).astype(np.float32)
    out_pil = np.array(ImageOps.solarize(impil, 70))[:, :, ::-1].astype(np.float32)
    print('solarize')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    n_test = 1000
    t1 = time.time()
    for i in range(n_test):
        out_cv = color_func(imcv, 0.6).astype(np.float32).astype(np.float32)[:, :, ::-1]
    t2 = time.time()
    for i in range(n_test):
        out_pil = np.array(ImageEnhance.Color(impil).enhance(0.6)).astype(np.float32)
    t3 = time.time()
    print('color')
    print('cv', t2 - t1)
    print('pil', t3 - t2)
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = contrast_func(imcv, 0.6).astype(np.float32).astype(np.float32)[:, :, ::-1]
    out_pil = np.array(ImageEnhance.Contrast(impil).enhance(0.6)).astype(np.float32)
    print('contrast')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = brightness_func(imcv, 0.6).astype(np.float32).astype(np.float32)[:, :, ::-1]
    out_pil = np.array(ImageEnhance.Brightness(impil).enhance(0.6)).astype(np.float32)
    print('brightness')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = shear_x_func(imcv, -0.2)
    cv2.imwrite('out_cv2.jpg', out_cv)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0.2, 0, 0, 1, 0), Image.BICUBIC, fillcolor=0)
    out_pil.save('out_pil.jpg')
    out_pil = np.array(out_pil).astype(np.float32)
    print('shear_x')
    print(out_pil[30:35, 40:45, 0])
    print(out_cv[30:35, 40:45, 0])
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = shear_y_func(imcv, 0.2)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0, 0, -0.2, 1, 0), Image.BICUBIC, fillcolor=0)
    out_pil = np.array(out_pil).astype(np.float32)
    print('shear_y')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = translate_x_func(imcv, 4)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0, 4, 0, 1, 0), Image.BICUBIC, fillcolor=0)
    out_pil = np.array(out_pil).astype(np.float32)
    print('translate_x')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    impil = Image.open(pth)
    imcv = cv2.imread(pth)
    out_cv = translate_y_func(imcv, 0.1)
    out_cv = out_cv.astype(np.float32)[:, :, ::-1]
    offset = int(0.1 * impil.size[1])
    out_pil = impil.transform(impil.size, Image.AFFINE, (1, 0, 0, 0, 1, offset), Image.BICUBIC, fillcolor=0)
    out_pil = np.array(out_pil).astype(np.float32)
    print('translate_y')
    print(np.max(np.abs(out_pil - out_cv)))
    print(np.min(np.abs(out_pil - out_cv)))

    ##
    pth = './pic.jpg'
    imcv = cv2.imread(pth)
    out_cv = cutout_func(imcv, 16, (128, 128, 128))
    cv2.imwrite('out_cv2.jpg', out_cv)
