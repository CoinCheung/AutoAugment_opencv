
# AutoAugment-Opencv

My implementation of [AutoAugmentation](https://arxiv.org/abs/1805.09501), [RandomAugmentation](https://arxiv.org/abs/1909.13719) and [AutoAugmentation for detection](https://arxiv.org/abs/1906.11172) with cv2.


<p align='center'>
<img src='pic.jpg'>
</p>

You can download this repo to your project:
```
    cd /your/project
    git clone https://github.com/CoinCheung/AutoAugment_opencv.git
``` 
and import it directly.

If your task is classification, you can do this:  
```python
    from AutoAugment_opencv.AA_classification  import AutoAugment, RandomAugment
    import cv2

    img = cv2.imread('./AutoAugment_opencv/pic.jpg')

    AA = AutoAugment()
    aa_auged = AA(img)

    RA = RandomAugment(N=2, M=9)
    ra_auged = RA(img)
```

If your task is object detection, you can use autoaugment for detection like this:   
```python
    from AutoAugment_opencv.AA_detection import PolicyV0, PolicyV1, PolicyV2, PolicyV3
    import xml.etree.ElementTree as ET
    import cv2
    import numpy as np

    xmlpth = './pascal_format.xml'
    jpgpth = './example_picture.jpg'

    img = cv2.imread(jpgpth)
    tree = ET.parse(xmlpth)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        bboxes.append(bbox)

    AA_det = PolicyV0() # PolicyV1, PolicyV2, PolicyV3 are results also worth trying in the paper
    aug_img, aug_bboxes = AA_det(img, bboxes)
```

This implementation only matches PIL when the cv2 is 3.4.2, if it is 4.1.2, there will be slight difference.

If you see any errors in this codebase, please be my guest to open an issue to correct me. Thanks !!

