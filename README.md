# [Single shot detection implementation](https://arxiv.org/abs/1512.02325)

## Objectives
This work is for learning purpose and the main goal is to recreate the results of SSD300 on Pascal VOC 2007 + 2012 dataset 
as published in the [paper](https://arxiv.org/abs/1512.02325).

## Illustrations
#### Learning curve
#### Results
#### Demo

## Dependencies
* Python 3
* PyTorch 1.6.0
* OpenCV 4.3.0
* albumentations 0.4.6

## Highlights
* Many thanks to a very [detailed tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) by sgrvinod
* Data preparation: like in the paper except for the 21503 training images was splited in to train and val set (80-20) so just 17k images left for training.
Also, all .xml annotation files was parsed in to one big json file.
* All the training took place in Google Colab runtime so big appreciation to Google for the generous offer of free GPUs
* Tried mixed precision O1 level on Google Colab's Tesla T4 GPU but no significant improvement in training speed (improvement of x1.1).

## To do
* Replace the backbone with something else like Resnet-50, Denset-201, SE-ResneXt-101, EfficientNet
* [Feature Pyramid Networks (FPN)](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
* [Focal loss](https://arxiv.org/abs/1708.02002v2)
* Experiment with more augmentations

## Pretrained

## Reference
