import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from math import sqrt


class SSD300(nn.Module):
    def __init__(self, n_classes, ):
        super(SSDModel, self).__init__()
        self.n_classes = n_classes
        
        # VGG16 conv layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)    # 300x300
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)           # 150x150

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # 75x75

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # 38x38

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                # 19x19

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)      # 19x19        
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)            # 19x19
        
        self.conv6 = nn.Conv2d(512,  1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)             # 19x19
        
        # Extra features layers
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 10x10
        
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 5x5
        
        self.conv10_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)  # 3x3
        
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)  # 1x1
        
        # Detection layers 
        self.det_conv4_3 = nn.Conv2d(512, 4*(4+n_classes), kernel_size=3, padding=1)
        self.det_conv7 = nn.Conv2d(1024, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_conv8_2 = nn.Conv2d(512, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_conv9_2 = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_conv10_2 = nn.Conv2d(256, 4*(4+n_classes), kernel_size=3, padding=1)
        self.det_conv11_2 = nn.Conv2d(256, 4*(4+n_classes), kernel_size=3, padding=1)
        
        self.init_weights()


    def init_weights():
        state_dict = self.state_dict()
        layer_names = list(state_dict.keys())
        
        vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        vgg16 = torch.hub.load_state_dict_from_url(vgg16_url, model_dir='models/')
        vgg16_layer_names = list(vgg16.keys())
            
        # Load from conv1_1 .. conv5_3
        for i, layer_name in enumerate(layer_names[0:26]):
            state_dict[layer_name] = vgg16[vgg16_layer_names[i]]
            
        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)
        
        # Init extra conv and clf layers
        for layer_name in layer_names[30:]:
            if layer_name[-4:] == 'bias':
                nn.init.zero_(state_dict[layer_name])
            elif layer_name[-6:] == 'weight':
                nn.init.xavier_uniform_(state_dict[layer_name])
            else:
                assert False
        
        self.load_state_dict(state_dict)
        
    
    def forward(self, x):
        '''
        x: tensor (n, 3, 300, 300)
        '''
        n = x.size(0)
        
        # VGG16 Feature extraction
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        fm38 = x
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        fm19 = x
        
        # Extra feature extraction
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        fm10 = x
        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x))
        fm5 = x
        x = F.relu(self.conv10_1(x))
        x = F.relu(self.conv10_2(x))
        fm3 = x
        x = F.relu(self.conv11_1(x))
        x = F.relu(self.conv11_2(x))
        fm1 = x

        # Detection
        det_fm38 = F.relu(self.det_conv4_3(fm38))
        det_fm38 = det_fm38.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4*(4+n_classes) )
        det_fm38 = det_fm38.view(n, -1, 4+n_classes)  # (N, 5776, 4+n_classes), there are a total 5776 boxes on this feature map
        
        det_fm19 = F.relu(self.det_conv7(fm19))
        det_fm19 = det_fm19.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6*(4+n_classes) )
        det_fm19 = det_fm19.view(n, -1, 4+n_classes)  # (N, 2166, 4+n_classes), there are a total 2166 boxes on this feature map
        
        det_fm10 = F.relu(self.det_conv8_2(fm10))
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6*(4+n_classes) )
        det_fm10 = det_fm10.view(n, -1, 4+n_classes)  # (N, 600, 4+n_classes), there are a total 600 boxes on this feature map
        
        det_fm5  = F.relu(self.det_conv9_2(fm5))
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6*(4+n_classes) )
        det_fm5 = det_fm5.view(n, -1, 4+n_classes)  # (N, 150, 4+n_classes), there are a total 150 boxes on this feature map
        
        det_fm3 = F.relu(self.det_conv10_2(fm3))
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4*(4+n_classes) )
        det_fm3 = det_fm3.view(n, -1, 4+n_classes)  # (N, 36, 4+n_classes), there are a total 36 boxes on this feature map
        
        det_fm1 = F.relu(self.det_conv11_2(fm1))
        det_fm1 = det_fm1.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4*(4+n_classes) )
        det_fm1 = det_fm1.view(n, -1, 4+n_classes)  # (N, 4, 4+n_classes), there are a total 4 boxes on this feature map
        
        detection = torch.cat([det_fm38, det_fm19, det_fm10, det_fm5, det_fm3, det_fm1], dim=1)  # (N, 8732, 4+n_classes)
        offsets, class_scores = torch.split(detection, [4,self.n_classes], dim=2)
        
        return offsets, class_scores
    
    
    def get_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        prior_boxes = []
        
        for k, fmap in enumerate(fmaps):
            dim = fmap_dims[fmap]
            for i in range(dim):
                for j in range(dim):
                    cx = (j + 0.5) * dim
                    cy = (i + 0.5) * dim
                    
                    s = obj_scales[fmap]
                    for ratio in aspect_ratios[fmap]:
                        w = s * sqrt(ratio)
                        h = s / sqrt(ratio)
                        prior_boxes.append([cx, cy, w, h])
                        
                    # additional prior box:
                    if fmap > 1:
                        additional_scale = sqrt(s * obj_scales[fmaps[k + 1]])
                    else:
                        additional_scale = 1
                    prior_boxes.append(cx, cy, additional_scale, additional_scale)
        
        prior_boxes = torch.Tensor(prior_boxes, dtype=torch.float)
        prior_boxes.clamp_(min=0, max=1)
        assert prior_boxes.shape == (8732,4)
        
        return prior_boxes

    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decode the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        
        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
    
    


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        '''
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors (n_objects, 4)
        :param labels: true object labels, a list of N tensors (n_objects, 1)
        :return: multibox loss, a scalar
        '''
        
        # matching boxes
        
        
        # bbox loss
        
        
        # classification loss
        
        
        
        #
        loss = 
        
        
        return loss
