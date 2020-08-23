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
        self.img_size = img_size
        self.transform = SSDTransform(size=img_size)
        self.augment = SSDAugmentation(size=img_size) if augment else None
        
        self.img_paths = list()
        self.targets = list()
        
        with open(os.path.join(self.root, json_files[0]), 'r') as f:
            self.img_paths = json.load(f)
        with open(os.path.join(self.root, json_files[1]), 'r') as f:
            self.targets = json.load(f)
        assert len(self.img_paths) == len(self.targets)
        
        if not self.keep_difficult:
            self.remove_difficult_objs()
    
    def __len__(self):
        return len(self.img_paths)
    
    def remove_difficult_objs(self):
        for target in self.targets:
            boxes = target['boxes']
            labels = target['labels']
            difficulties = target['difficulties']
            
            # remove difficult objects if not keep_difficult
            boxes =  [boxes[i] for i in range(len(boxes)) if not difficulties[i]]
            labels = [labels[i] for i in range(len(labels)) if not difficulties[i]]
    
    def __getitem__(self, index):
        img = self.read_img(index)
        target = self.targets[index]
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        
        if self.augment is not None:
            img, boxes, labels = self.augment(img, boxes, labels)
            if len(boxes) < 1: 
                # This is the case when albumentations crop or shrink all the objects and discard all of them
                # So in this case(~0.5%), no augmentation will be done.
                img, boxes, labels = self.read_img(index), target['boxes'].copy(), target['labels'].copy()
            
        img, boxes, labels = self.transform(img, boxes, labels)
            
        #save_aug(img, boxes, labels, os.path.basename(self.img_paths[index]))
        # to tensor
        img = torch.Tensor(img.transpose((2,0,1)))
        boxes = rescale_coordinates(boxes, h=img.size(1), w=img.size(2))
        labels = torch.IntTensor(labels)
        
        return img, boxes, labels
    
    def read_img(self, index):
        ''' read the image and convert to RGB '''
        return cv2.cvtColor(cv2.imread(self.img_paths[index]), cv2.COLOR_BGR2RGB)


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

    for imgs, boxes, labels in batch:
        batch_imgs.append(imgs)
        batch_boxes.append(boxes)
        batch_labels.append(labels)

    batch_imgs = torch.stack(batch_imgs, dim=0)

    return batch_imgs, batch_boxes, batch_labels  # tensor(N, 3, 300, 300) and 2 lists of N tensors each


if __name__=="__main__":
    torch.manual_seed(42)
    trainset = VOCDataset(data_folder='datasets/', json_files=('TRAIN_images.json', 'TRAIN_objects.json'), augment=True)
    valset = VOCDataset(data_folder='datasets/', json_files=('VAL_images.json', 'VAL_objects.json'))
    
    dataloaders = dict(
        train = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True, num_workers=2),
        val = DataLoader(valset, batch_size=32, collate_fn=collate_fn, shuffle=False, num_workers=2),
    )
    
    import time
    start_time = time.time()
    for step, (imgs, boxes, labels) in enumerate(dataloaders['train']):
        #print(imgs.shape)
        if step % 50 == 0: print('step:', step, '- loaded imgs:', step*32)
    print(time.time() - start_time)
    
    start_time = time.time()
    for step, (imgs, boxes, labels) in enumerate(dataloaders['val']):
        if step % 10 == 0: print('step:', step, '- loaded imgs:', step*32)
    print(time.time() - start_time)
