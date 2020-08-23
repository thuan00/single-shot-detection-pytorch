import albumentations as A
import numpy as np
import cv2
import random

#-----------------------------
# Transforms during test time
#-----------------------------
class SSDInputTransform(object):
    ''' Input: image: numpy.ndarray in uint8
        Output: numpy.ndarray in float32 and padding
    '''
    def __init__(self, size=300):
        self.transform = A.Compose([
            A.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
            A.Normalize(),
        ])
        
    def __call__(self, image):
        # padding for slim image(h>w), so that the aspect ratio of object is not changed too much
        # for wide image(h<w), it's acceptable to resize normally
        padding = None
        h, w = image.shape[0:2]
        if h > w:
            padding = int((h-w)/2)
            padded_img = np.full((h,h,3), (123, 117, 104), dtype='uint8')
            padded_img[:, padding:padding+w, :] = image
            image = padded_img
        # perform size and pixel value normalization
        transformed = self.transform(image=image)
        return transformed['image'], padding


#---------------------------------
# Transforms during training time
#---------------------------------
class SSDTransform(object):
    def __init__(self, size=300):
        self.transform = A.Compose([
            A.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
            #A.Normalize(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
    def __call__(self, image, boxes, labels):
        h, w = image.shape[0:2]
        if h > w:
            image, boxes = pad(image, boxes, h, h, mode=RANDOM_PAD)
        # perform regular transform
        transformed = self.transform(image=image, bboxes=boxes, labels=labels)
        return transformed['image'], transformed['bboxes'], transformed['labels']

    
class SSDAugmentation(object):
    def __init__(self, size=300):
        self.output_size = size
        self.typical_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(var_limit=12.0, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        self.random_crop = [None]*5  #I placed 5 mode of random crop to replicate real randomness in croping
        self.random_crop[0] = A.Compose([
            A.RandomCrop(width=size, height=size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5, min_area=100))
        
        self.random_crop[1] = A.Compose([
            A.RandomCrop(width=round(size*1.25), height=size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.4, min_area=100))
        
        self.random_crop[2] = A.Compose([
            A.RandomCrop(width=round(size*1.25), height=round(size*0.75)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.4, min_area=100))
        
        self.random_crop[3] = A.Compose([
            A.RandomCrop(width=round(size*1.5), height=size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3, min_area=100))
        
        self.random_crop[4] = A.Compose([
            A.RandomCrop(width=round(size*1.2), height=round(size*1.2)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3, min_area=100))
        
        self.zoomout_bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=150)
        
    def __call__(self, image, boxes, labels):
        augmented = self.typical_augmentations(image=image, bboxes=boxes, labels=labels)
        image, boxes, labels = augmented['image'], augmented['bboxes'], augmented['labels']
        
        p = random.random()
        
        #50% Zoom out
        if p > 0.5:
            h,w = image.shape[0:2]
            scale_h = self.output_size/h
            scale_w = self.output_size/w if (w < h) else scale_h*h/w
            zoomout_scale = random.uniform(0.4, 0.9)
            random_resize = A.Compose([
                A.Resize(width=round(w*scale_w*zoomout_scale), height=round(h*scale_h*zoomout_scale)),
            ], bbox_params=self.zoomout_bbox_params)
            zoomed_out = random_resize(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = zoomed_out['image'], zoomed_out['bboxes'], zoomed_out['labels']
            image, boxes = pad(image, boxes, self.output_size, self.output_size, mode=RANDOM_PAD)
            
        #25% Random crop
        if p < 0.25:
            # choosing crop mode
            random_crop = self.random_crop[random.randint(0, len(self.random_crop)-1)]
            
            # If requested crop size larger than the image size, pad the image before cropping
            target_h, target_w = random_crop.transforms[0].height, random_crop.transforms[0].width
            image_h, image_w = image.shape[0:2]
            if image_h < target_h or image_w < target_w:
                image, boxes = pad(image, boxes, target_h, target_w, slim_to_square=False)
            
            # Crop
            cropped = random_crop(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = cropped['image'], cropped['bboxes'], cropped['labels']
        
        #25% no crop, no pad, just typical augmentations
        return image, boxes, labels


# Padding mode
EQUAL_PAD = 0 # all sides of the image will be padded equally
RANDOM_PAD = 1 # all sides of the image will be padded randomly

def pad(image, boxes, target_h, target_w, mode=EQUAL_PAD, slim_to_square=True):
        ''' Expand the image so that the output shape equals or exceeds the target shape
            Padding is filled with ImageNet mean
        Params:
            image: image, ndarray
            boxes: coordinates of bounding boxes of the image, 2d list
            target_h: the height of the padded image should exceed target_h
            target_w: the width of the padded image should exceed target_w
            mode: int, modes described above
            slim_to_square: check the image is slim, do padding so that it is square
        Return: 
            padded image and its bboxes
        '''
        h, w = image.shape[0:2]
        if h > w and slim_to_square:
            target_w = target_h
        
        padding_h = max(0, target_h - h)
        padding_w = max(0, target_w - w)
        
        if mode == EQUAL_PAD:
            i = round(padding_h/2)
            j = round(padding_w/2)
        elif mode == RANDOM_PAD:
            i = round(random.random()*padding_h)
            j = round(random.random()*padding_w)
        else:
            raise RuntimeError("Unrecognised mode in pad function")

        padded_img = np.full((h+padding_h, w+padding_w, 3), (123, 117, 104), dtype='uint8')
        padded_img[i:i+h, j:j+w, :] = image
        
        for b in range(len(boxes)):
            x1, y1, x2, y2 = boxes[b]
            boxes[b] = (x1+j, y1+i, x2+j, y2+i)
        
        return padded_img, boxes
    
