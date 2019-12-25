
# AutoAugment-Opencv

My implementation of AutoAugmentation and RandomAugmentation with cv2.

You can download this repo to your project:
```
    cd /your/project
    git clone https://github.com/CoinCheung/AutoAugment-opencv.git
``` 
and use it directly:
```python
    from AutoAugment-opencv  import AutoAugment
    from AutoAugment-opencv  import RandomAugment

    import cv2
    img = cv2.imread('./AutoAugment-opencv/pic.jpg')

    AA = AutoAugment()
    aa_auged = AA(img)

    RA = RandomAugment(N=2, M=9)
    ra_auged = RA(img)
```

If you see any errors in this codebase, please be my guest to open an issue to correct me. Thanks !!

<p align='center'>
<img src='pic.jpg'>
</p>
