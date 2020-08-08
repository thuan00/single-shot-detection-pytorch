import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchvision.ops import nms
from utils import decimate, xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, find_jaccard_overlap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSD300(nn.Module):
    def __init__(self, n_classes, vgg16_dir, checkpoint):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.vgg16_dir = vgg16_dir
        
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
        
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
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
        
        if checkpoint == None:
            self.init_weights()
        else:
            self.load_state_dict(checkpoint['model'])
            
        self.priors_cxcy = self.get_prior_boxes()


    def init_weights(self):
        ''' Load pretrained VGG16 parameters for some first layers and initialize the rest
        '''
        state_dict = self.state_dict()
        layer_names = list(state_dict.keys())
        
        vgg16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        vgg16 = torch.hub.load_state_dict_from_url(vgg16_url, model_dir = self.vgg16_dir)
        vgg16_layer_names = list(vgg16.keys())
            
        # Load from conv1_1 .. conv5_3
        for i, layer_name in enumerate(layer_names[0:26]):
            state_dict[layer_name] = vgg16[vgg16_layer_names[i]]
            
        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = vgg16['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = vgg16['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = vgg16['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = vgg16['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)
        
        # Init extra conv and clf layers
        for layer_name in layer_names[30:]:
            if layer_name[-4:] == 'bias':
                nn.init.zeros_(state_dict[layer_name])
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
        box_size = 4 + self.n_classes  # each box has 25 values: 4 offset values and 21 class scores
        #
        det_fm38 = F.relu(self.det_conv4_3(fm38))
        det_fm38 = det_fm38.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4*box_size )
        det_fm38 = det_fm38.view(n, -1, box_size)  # (N, 5776, box_size), there are a total 5776 boxes on this feature map
        
        det_fm19 = F.relu(self.det_conv7(fm19))
        det_fm19 = det_fm19.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6*box_size )
        det_fm19 = det_fm19.view(n, -1, box_size)  # (N, 2166, box_size), there are a total 2166 boxes on this feature map
        
        det_fm10 = F.relu(self.det_conv8_2(fm10))
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6*box_size )
        det_fm10 = det_fm10.view(n, -1, box_size)  # (N, 600, box_size), there are a total 600 boxes on this feature map
        
        det_fm5  = F.relu(self.det_conv9_2(fm5))
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6*box_size )
        det_fm5 = det_fm5.view(n, -1, box_size)  # (N, 150, box_size), there are a total 150 boxes on this feature map
        
        det_fm3 = F.relu(self.det_conv10_2(fm3))
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4*box_size )
        det_fm3 = det_fm3.view(n, -1, box_size)  # (N, 36, box_size), there are a total 36 boxes on this feature map
        
        det_fm1 = F.relu(self.det_conv11_2(fm1))
        det_fm1 = det_fm1.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4*box_size )
        det_fm1 = det_fm1.view(n, -1, box_size)  # (N, 4, box_size), there are a total 4 boxes on this feature map
        
        detection = torch.cat([det_fm38, det_fm19, det_fm10, det_fm5, det_fm3, det_fm1], dim=1)  # (N, 8732, box_size)
        offsets, class_scores = torch.split(detection, [4,self.n_classes], dim=2)
        
        return offsets, class_scores
    
    
    def get_prior_boxes(self):
        '''
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        Return: 
            prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        '''
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
                    cx = (j + 0.5) / dim
                    cy = (i + 0.5) / dim
                    
                    s = obj_scales[fmap]
                    for ratio in aspect_ratios[fmap]:
                        w = s * sqrt(ratio)
                        h = s / sqrt(ratio)
                        prior_boxes.append([cx, cy, w, h])
                        
                    # an additional prior box:
                    if dim > 1:
                        additional_scale = sqrt(s * obj_scales[fmaps[k + 1]])
                    else:
                        additional_scale = 1
                    prior_boxes.append([cx, cy, additional_scale, additional_scale])
        
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(min=0, max=1)
        assert prior_boxes.shape == (8732,4)
        
        return prior_boxes

    
    def detect_objects(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold):
        '''
        Decode the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        Notes: 
            The use of torch.no_grad() or torch.set_grad_enabled(False) before calling this method is recommended
            Since this method is only called for evaluation and inference purpose so there is no need for calculating grads
        
        Params:
            predicted_offsets: predicted offsets w.r.t the 8732 prior boxes, (gcxgcy), a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            score_threshold: minimum threshold for a box to be considered a match for a certain class
            iou_threshold: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        Return: 
            detections: (boxes, labels, and scores), lists of N tensors
            boxes: N (n_boxes, 4)
            labels: N (n_boxes,)
            scores: N (n_boxes,)
        '''
        boxes = list()
        labels = list()
        scores = list()
        
        N, n_priors = predicted_offsets.shape[0:2]
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
        
        # for each box, find the largest score and the class_id with respect to it
        class_scores, class_ids = predicted_scores.max(dim=2) # (N, 8732) and (N, 8732)
        
        # for each unprocessed predictions in the batch:
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            
            # filter out boxes that are not qualified, that were predicted as background or with low confidence score
            qualify_mask = (class_ids[i] != 0) & (class_scores[i] > score_threshold) # (8732)
            qualified_boxes = predicted_offsets[i][qualify_mask]  # (n_qualified_boxes, 4)
            qualified_boxes_class = class_ids[i][qualify_mask]    # (n_qualified_boxes)
            qualified_boxes_score = class_scores[i][qualify_mask] # (n_qualified_boxes)
            
            if len(qualified_boxes) != 0:
                # convert to xy coordinates format
                qualified_boxes = cxcy_to_xy(gcxgcy_to_cxcy(qualified_boxes, self.priors_cxcy[qualify_mask])) # (n_qualified_boxes, 4)

                # Non-max suppression
                for class_i in qualified_boxes_class.unique(sorted=False).tolist():
                    class_mask = qualified_boxes_class == class_i

                    boxes_class_i = qualified_boxes[class_mask]
                    boxes_score_class_i = qualified_boxes_score[class_mask]

                    final_box_ids = nms(boxes_class_i, boxes_score_class_i, iou_threshold)  # (n_final_boxes,)

                    boxes_i.extend(boxes_class_i[final_box_ids].tolist())
                    labels_i.extend([class_i]*len(final_box_ids))
                    scores_i.extend(boxes_score_class_i[final_box_ids].tolist())
        
            boxes.append(torch.FloatTensor(boxes_i).to(device))
            labels.append(torch.LongTensor(labels_i).to(device))
            scores.append(torch.FloatTensor(scores_i).to(device))
        
        return boxes, labels, scores 



class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predicted_offsets, predicted_scores, boxes, labels):
        '''
        Params:
            predicted_offsets: predicted offsets w.r.t the 8732 prior boxes, (gcxgcy), a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            boxes: true  object bounding boxes in boundary coordinates, (xy), a list of N tensors: (n_objects, 4)
            labels: true object labels, a list of N tensors: (n_objects,)
        Return: 
            multibox loss, a scalar
        '''
        N = predicted_offsets.shape[0]
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        truth_offsets = torch.zeros((N, n_priors, 4), dtype=torch.float).to(device)
        truth_classes = torch.zeros((N, n_priors), dtype=torch.long).to(device)
        
        # Matching ground truth boxes
        # for each image
        for i in range(N):
            n_objects = labels[i].shape[0]
            
            overlap = find_jaccard_overlap(self.priors_xy, boxes[i]) #(n_priors, n_boxes)
            
            # for each prior, find the max iou and the coresponding object id
            prior_iou, prior_obj = overlap.max(dim=1) #(n_priors)
            
            # for each object, find the most suited prior id
            _, object_prior = overlap.max(dim=0) #(n_objects)
            # for each object, assign its most suited prior with object id 
            #for j in range(n_objects): prior_obj[object_prior[j]] = j
            prior_obj[object_prior] = torch.LongTensor(range(n_objects)).to(device)
            # for each object, assign its most suited prior with hight iou to ensure it qualifies the thresholding 
            prior_iou[object_prior] = 1.
            
            # match bbox coordinates
            boxes_xy = boxes[i][prior_obj] # (8732, 4)
            
            # match prior class
            prior_class = labels[i][prior_obj]  # (8732)
            # thresholding: assign prior with iou < threshold to the class 0: background
            prior_class[prior_iou < self.threshold] = 0
            
            # save in the truth tensors
            truth_offsets[i,:,:] = cxcy_to_gcxgcy(xy_to_cxcy(boxes_xy), self.priors_cxcy)
            truth_classes[i,:] = prior_class
        
        # Now we have truth_offsets, truth_classes and predicted_offsets, predicted_scores, we can now calculate the loss
        
        positive_priors = (truth_classes != 0) #(N,8732)
        n_positives = positive_priors.sum(dim=1)  # (N)
        
        # Calculating loss = alpha*loc_loss + conf_loss
        # loc_loss: localization loss
        loc_loss = self.smooth_l1(predicted_offsets[positive_priors], truth_offsets[positive_priors])
        
        # Confidence loss
        full_conf_loss = self.cross_entropy(predicted_scores.view(-1, n_classes), truth_classes.view(-1)) #(N*n_priors)
        full_conf_loss = full_conf_loss.view(N, n_priors)
        # However there is a huge unbalance between positive and negative priors so we only take the loss of the hard negative priors

        # Hard negative mining
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)
        conf_loss_hard_neg = 0
        # accummulate conf_loss_hard_neg for each sample in batch
        for i in range(N):
            conf_loss_neg,_ = full_conf_loss[i][~positive_priors[i]].sort(dim=0, descending=True) # (1-n_positives)
            conf_loss_hard_neg = conf_loss_hard_neg + conf_loss_neg[0:n_hard_negatives[i]].sum()
        
        conf_loss = (full_conf_loss[positive_priors].sum() + conf_loss_hard_neg) / n_positives.sum()

        return self.alpha * loc_loss + conf_loss


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    MySSD300 = SSD300(n_classes = 21)
    loss_func = MultiBoxLoss(priors_cxcy = MySSD300.priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.)
    
    predicted_offsets = torch.randn((2,8732,4))
    predicted_scores = torch.randn((2,8732,21))
    
    boxes = [torch.Tensor([[0.1, 0.2, 0.3, 0.4]]), torch.Tensor([[0.1, 0.2, 0.3, 0.4]])]
    labels = [torch.LongTensor([2]), torch.LongTensor([3])]
    
    # test loss function
    loss = loss_func.forward(predicted_offsets, predicted_scores, boxes, labels)
    print(loss.item())
    
    # test detect objects
    boxes, labels, scores = MySSD300.detect_objects(predicted_offsets, predicted_scores, score_threshold=0.6, iou_threshold=0.5)
    breakpoint()
    