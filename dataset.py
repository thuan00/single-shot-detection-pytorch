import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from aug import SSDAugmentation, SSDTransform


class VOCDataset(Dataset):
    def __init__(self, data_folder, json_files, augment=False):
        super(VOCDataset, self).__init__()
        self.root = data_folder
        self.transform = SSDTransform(size=300)
        self.augment = SSDAugmentation(size=300) if augment else None
        
        self.img_paths = list()
        self.targets = list()
        
        with open(os.path.join(self.root, json_files[0]), 'r') as f:
            self.img_paths = json.load(f)
        with open(os.path.join(self.root, json_files[1]), 'r') as f:
            self.targets = json.load(f)
        assert len(self.img_paths) == len(self.targets)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        boxes = target['boxes']
        labels = target['labels']
        
        if self.augment is not None:
            img, boxes, labels = self.augment(img, boxes.copy(), labels.copy())
            
        img, boxes, labels = self.transform(img, boxes, labels)
        
        # This is the case when albumentations crop or shrink all the objects and discard those with area <50px,
        # so we need to re-augment with the no_crop_pad option that ensure there is at least one obj in the image
        while self.augment and len(boxes) < 1: 
            #print('*')
            img = cv2.imread(self.img_paths[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, boxes, labels = self.augment(img, target['boxes'].copy(), target['labels'].copy(), no_crop_pad=True)
            img, boxes, labels = self.transform(img, boxes, labels)
            
        # to tensor
        img = torch.Tensor(img.transpose((2,0,1)))
        boxes = torch.from_numpy(bbox_transform(boxes, h=300, w=300))
        labels = torch.IntTensor(labels)
        diffs = torch.IntTensor(target['difficulties'])
        
        return img, boxes, labels, diffs



def validate_aug(img, boxes, img_id):
    #validate_aug(img, boxes, self.img_paths[index][-15:-4])
    if '/' in img_id:
        img_id = img_id[5:]
    # validate augmentation
    im_cp = img.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min, y_min, x_max, y_max = round(x_min), round(y_min), round(x_max), round(y_max)
        im_cp = cv2.rectangle(im_cp, (x_min,y_min), (x_max,y_max), (0, 255, 0), 1)
    im_cp = cv2.cvtColor(im_cp, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'datasets/augmented_data/3/'+img_id+'.png', im_cp)

def bbox_transform(boxes, h, w):
    ''' Normalize the bbox coordinates to 0-1 format
    boxes: list shape(n_boxes, 4)
    return: np
    '''
    boxes = np.array(boxes, dtype='float32')
    boxes[:,0] /= w
    boxes[:,1] /= h
    boxes[:,2] /= w
    boxes[:,3] /= h
    return boxes

def collate_fn(batch):
    """ Explaination
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.

    Param: batch: an iterable of N sets from __getitem__()
    Return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    batch_imgs = list()
    batch_boxes = list()
    batch_labels = list()
    batch_diffs = list()

    for imgs, boxes, labels, diffs in batch:
        batch_imgs.append(imgs)
        batch_boxes.append(boxes)
        batch_labels.append(labels)
        batch_diffs.append(diffs)

    batch_imgs = torch.stack(batch_imgs, dim=0)

    return batch_imgs, batch_boxes, batch_labels, batch_diffs  # tensor(N, 3, 300, 300) and 2 lists of N tensors each


if __name__=="__main__":
    torch.manual_seed(42)
    trainset = VOCDataset(data_folder='datasets/', json_files=('TRAIN_images.json', 'TRAIN_objects.json'), augment=True)
    valset = VOCDataset(data_folder='datasets/', json_files=('VAL_images.json', 'VAL_objects.json'))
    dataloaders = dict(
        train = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True, num_workers=4),
        val = DataLoader(valset, batch_size=64, collate_fn=collate_fn, shuffle=False, num_workers=4),
    )
    
    for step, (imgs, boxes, labels,_) in enumerate(dataloaders['train']):
        #print(imgs.shape)
        if step % 50 == 0: print('step:', step, '- loaded imgs:', step*32)
    
    for step, (imgs, boxes, labels,_) in enumerate(dataloaders['val']):
        if step % 50 == 0: print('step:', step, '- loaded imgs:', step*64)