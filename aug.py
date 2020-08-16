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
            A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR),
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
            A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=50))
        
    def __call__(self, image, boxes, labels):
        h, w = image.shape[0:2]
        if h > w:
            padding = random.randint(0, h-w)
            padded_img = np.full((h,h,3), (123, 117, 104), dtype='uint8')
            padded_img[:, padding:padding+w, :] = image
            for b in range(len(boxes)):
                x1, y1, x2, y2 = boxes[b]
                boxes[b] = (x1+padding, y1, x2+padding, y2)
            image = padded_img
        # perform regular transform
        transformed = self.transform(image=image, bboxes=boxes, labels=labels)
        return transformed['image'], transformed['bboxes'], transformed['labels']

    
class SSDAugmentation(object):
    def __init__(self, size=300):
        self.random_crop = [None]*4
        self.random_crop[0] = A.Compose([
            A.RandomCrop(width=size, height=size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5))
        
        self.random_crop[1] = A.Compose([
            A.RandomCrop(width=round(size*1.3), height=size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.4))
        
        self.random_crop[2] = A.Compose([
            A.RandomCrop(width=round(size*1.5), height=size),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
        
        self.random_crop[3] = A.Compose([
            A.RandomCrop(width=round(size*1.2), height=round(size*1.2)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
        
        self.final_augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(var_limit=12.0, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
    def __call__(self, image, boxes, labels, no_crop_pad=False):
        p = random.random()
        if no_crop_pad: p = 0.3 #avoid crop and pad, just perform regular augmentations
        
        #50% Random pad
        if p > 0.5: 
            image, boxes = self.random_pad(image, boxes)
            
        #25% Random crop
        if p < 0.25:
            random_crop = self.random_crop[random.randint(0, len(self.random_crop)-1)]
            target_h, target_w = random_crop.transforms[0].height, random_crop.transforms[0].width
            h, w = image.shape[0:2]
            if h < target_h or w < target_w: # If requested crop size larger than the image size, pad
                image, boxes = self.pad(image, boxes, target_h, target_w)
            # ensure there is at least one obj in the image
            n_iter = 0
            while n_iter < 5:
                cropped = random_crop(image=image.copy(), bboxes=boxes.copy(), labels=labels.copy())
                n_iter += 1
                if len(cropped['bboxes']) > 0: 
                    image, boxes, labels = cropped['image'], cropped['bboxes'], cropped['labels']
                    break
                    
        #25% no crop, no pad, just regular augmentations
        augmented = self.final_augment(image=image, bboxes=boxes, labels=labels)
        return augmented['image'], augmented['bboxes'], augmented['labels']
    
    def random_pad(self, image, boxes):
        ''' Expand the image and fill the border with ImageNet mean
            If the image is slim, also do padding so that it is square
        '''
        h, w = image.shape[0:2]
        scale = random.uniform(1.1,2.2) - 1
        padding_h = round(h * scale)
        padding_w = (h+padding_h - w) if (h > w) else round(w * scale)
        i = round(random.random()*padding_h)
        j = round(random.random()*padding_w)

        padded_img = np.full((h+padding_h, w+padding_w, 3), (123, 117, 104), dtype='uint8')
        padded_img[i:i+h, j:j+w, :] = image
        
        for b in range(len(boxes)):
            x1, y1, x2, y2 = boxes[b]
            boxes[b] = (x1+j, y1+i, x2+j, y2+i)

        return padded_img, boxes
    
    def pad(self, image, boxes, target_h, target_w):
        ''' equally padding all sides so that the image's shape >= the target shape'''
        h, w = image.shape[0:2]
        padding_h = max(0, target_h - h)
        padding_w = max(0, target_w - w)
        i = round(padding_h/2)
        j = round(padding_w/2)

        padded_img = np.full((h+padding_h, w+padding_w, 3), (123, 117, 104), dtype='uint8')
        padded_img[i:i+h, j:j+w, :] = image
        
        for b in range(len(boxes)):
            x1, y1, x2, y2 = boxes[b]
            boxes[b] = (x1+j, y1+i, x2+j, y2+i)
        
        return padded_img, boxes


    
def visualize_bboxes(img, bboxes, thickness=1):
    """Visualizes a single bounding box on the image"""
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = round(x_min), round(y_min), round(x_max), round(y_max)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0,255,0), thickness=thickness)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    from utils import parse_annotation
    
    img_path = 'datasets/s/2008_000424.jpg'
    anno_path = 'datasets/s/2008_000424.xml'
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target = parse_annotation(anno_path)
    
    transform = A.Compose([
        #A.RandomCropNearBBox(max_part_shift=0.3, p=1.0)
        #A.Resize(height=300, width=300, interpolation=cv2.INTER_LINEAR)
        A.RandomCrop(width=300, height=300, p=0.5)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.25, min_area=200))
    
    transformed = transform(image=img, bboxes=target['boxes'], labels=target['labels'], cropping_bbox=target['boxes'][2])
    
    #breakpoint()
    img = visualize_bboxes(transformed['image'], transformed['bboxes'])
    
    cv2.imshow('img', img)
    cv2.waitKey()
    
