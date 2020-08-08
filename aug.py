import albumentations as A
import numpy as np
import cv2
import torch
from torchvision import transforms


class SSDTransform(object):
    def __init__(self, input_size=300):
        self.transform = A.Compose([
            A.Resize(   ) ,
            A.RandomCrop(width=, height=),
        ], bbox_params=A.BboxParams(format='pascal_voc'))
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, images, boxes, labels):
        
        
        return images, boxes, labels

class SSDAugmentation(object):
    def __init__(self, input_size=300):
        self.augment = A.Compose([
            A.RandomCrop(width=, height=),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.2),
            A.CLAHE(p=0.2),  
        ], bbox_params=A.BboxParams(format='coco'))
        
        self.transform = SSDTransform(input_size)
        
    def __call__(self, images, boxes, labels):
        

        return 
    


if __name__ == "__main__":
    img = cv2.imread("id.png")
    imgs = []
    for i in range(100):
        imgs.append(img)
        
    transform = SSDAugmentation()
    newDatas = transform(imgs)
    print(len(newDatas))

    for i, img in enumerate(newDatas):
        cv2.imwrite("__"+str(i)+".png", img)
