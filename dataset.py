import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from aug import SSDAugmentation, SSDTransform

class VOCDataset(Dataset):
    def __init__(self, data_folder, json_files, augment=False):
        self.root = data_folder
        self.transform = SSDTransform(input_size=300)
        self.augment = SSDAugmentation(input_size=300) if augment else None
        
        self.img_paths = list()
        self.targets = list()
        
        with open(os.path.join(self.root, json_files[0]), 'r') as f:
            self.img_paths = json.load(f)
        with open(os.path.join(self.root, json_files[1]), 'r') as f:
            self.targets = json.load(f)
        assert len(self.img_paths) = len(self.targets)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #
        target = self.targets[index]
        boxes = self.targets['boxes']
        labels = self.targets['labels']
        diffs = torch.IntTensor(target['difficulties'])
        
        # augmentation
        if self.augment not None:
            img, boxes, labels = self.augment(img, boxes, labels)
            
        # final transform
        img, boxes, labels = self.transform(img, boxes, labels)

        validate_aug(img, boxes)
        
        # to tensor & normalize
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)
        
        return img, boxes, labels, diffs
    
    def validate_aug(img, boxes):
        # validate augmentation
        im_cp = img.copy()
        boxes_cp = np.rint(boxes.copy()*300).astype('int')
        for box in boxes_cp.tolist():
            #breakpoint()
            x1, y1, x2, y2 = box
            im_cp = cv2.rectangle(im_cp, (x1,y1), (x2,y2), (0, 255, 0), 2)
        cv2.imwrite(f'datasets/augmented_data/'+self.img_paths[index][-15:-4]+'.png', im_cp)



def bbox_transform(boxes, h, w):
    ''' Normalize the bbox coordinates to 0-1 inplace
    boxes: np array shape(n_boxes, 4)
    '''
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
    trainset = VOCDataset(data_folder='datasets/', augmentation=True)
    valset = VOCDataset(data_folder='datasets/', augmentation=False)
    dataloaders = dict(
        train = DataLoader(trainset, batch_size=8, collate_fn=collate_fn, shuffle=True, num_workers=4),
        val = DataLoader(valset, batch_size=64, collate_fn=collate_fn, shuffle=False, num_workers=4),
    )
    
    for step, (imgs, boxes, labels,_) in enumerate(dataloaders['train']):
        #print(imgs.shape)
        if step == 19: break