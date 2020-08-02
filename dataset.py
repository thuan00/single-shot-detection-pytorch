import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from aug2 import SSDAugmentation


def VOCbbox_transform(boxes, h, w):
    ''' Normalize the bbox coordinates to 0-1 inplace
    boxes: np array shape(n_boxes, 4)
    '''
    boxes[:,0] /= w
    boxes[:,1] /= h
    boxes[:,2] /= w
    boxes[:,3] /= h

class VOCDataset(Dataset):
    def __init__(self, data_folder, is_trainset=True):
        self.root = data_folder
        self.is_trainset = is_trainset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = SSDAugmentation(size=300) if is_trainset else None
        
        self.img_paths = list()
        self.targets = list()
        
        if self.is_trainset:
            with open(os.path.join(self.root, 'TRAIN_images.json'), 'r') as f:
                self.img_paths = json.load(f)
            with open(os.path.join(self.root, 'TRAIN_objects.json'), 'r') as f:
                self.targets = json.load(f)
        else:# testset:
            with open(os.path.join(self.root, 'TEST_images.json'), 'r') as f:
                self.img_paths = json.load(f)
            with open(os.path.join(self.root, 'TEST_objects.json'), 'r') as f:
                self.targets = json.load(f)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        h, w = img.shape[0:2]
        
        target = self.targets[index]
        boxes = np.array(target['boxes'], dtype='float32')
        VOCbbox_transform(boxes, h, w)
        labels = np.array(target['labels'], dtype='int32')
        
        if self.augment is not None:
            img, boxes, labels = self.augment(img, boxes, labels)
        
        #to tensor & normalize
        img = img[:, :, (2,1,0)] #to RGB
        img = self.transform(img)
        
        return img, boxes, labels
    
    def collate_fn(self, batch):
        """ Explaination
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        batch_imgs = list()
        batch_boxes = list()
        batch_labels = list()
        
        for img, b, l, d in batch:
            batch_imgs.append(img)
            batch_boxes.append(b)
            batch_labels.append(l)
            
        images = torch.stack(images, dim=0)
        
        return batch_imgs, batch_boxes, batch_labels  # tensor(N, 3, 300, 300) and 2 lists of N tensors each

if __name__=="__main__":
    torch.manual_seed(42)
    trainset = VOCDataset(data_folder='datasets/', is_trainset=True)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    #testset = VOCDataset(data_folder='datasets/', is_trainset=False)

    '''
    dataloaders = dict(
        train = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0),
        val = DataLoader(valset, batch_size=64, shuffle=False, num_workers=0),
        test = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    )'''
    
    val = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    
    for img, b, l in val:
        print(img.shape)
        print(b)
        print(l)
        break