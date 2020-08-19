import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from aug import SSDAugmentation, SSDTransform
from utils import rescale_coordinates, save_aug


class VOCDataset(Dataset):
    def __init__(self, data_folder, json_files, augment=False, keep_difficult=False, img_size=300):
        super(VOCDataset, self).__init__()
        self.root = data_folder
        self.keep_difficult = keep_difficult
        self.transform = SSDTransform(size=img_size)
        self.augment = SSDAugmentation(size=img_size) if augment else None
        
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
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        difficulties = target['difficulties'].copy()
        
        # remove difficult objects if not keep_difficult
        if not self.keep_difficult:
            boxes =  [boxes[i] for i in range(len(boxes)) if not difficulties[i]]
            labels = [labels[i] for i in range(len(labels)) if not difficulties[i]]
            difficulties = [difficulties[i] for i in range(len(difficulties)) if not difficulties[i]]
        
        if self.augment is not None:
            img, boxes, labels = self.augment(img, boxes, labels)
            
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
        boxes = rescale_coordinates(boxes, h=img.size(1), w=img.size(2))
        labels = torch.IntTensor(labels)
        diffs = torch.IntTensor(difficulties)
        
        return img, boxes, labels, diffs


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
        if step == 9: break
        if step % 50 == 0: print('step:', step, '- loaded imgs:', step*32)
    
    #for step, (imgs, boxes, labels,_) in enumerate(dataloaders['val']):
    #    if step % 50 == 0: print('step:', step, '- loaded imgs:', step*64)