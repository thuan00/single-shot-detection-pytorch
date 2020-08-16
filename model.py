import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchvision.ops import nms
from utils import decimate, xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, find_jaccard_overlap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSD300(nn.Module):
    def __init__(self, n_classes, vgg16_dir=None, checkpoint=None):
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
            self.load_state_dict(checkpoint['model'].state_dict())
            
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
        det_fm38 = self.det_conv4_3(fm38)
        det_fm38 = det_fm38.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4*box_size )
        det_fm38 = det_fm38.view(n, -1, box_size)  # (N, 5776, box_size), there are a total 5776 boxes on this feature map
        
        det_fm19 = self.det_conv7(fm19)
        det_fm19 = det_fm19.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6*box_size )
        det_fm19 = det_fm19.view(n, -1, box_size)  # (N, 2166, box_size), there are a total 2166 boxes on this feature map
        
        det_fm10 = self.det_conv8_2(fm10)
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6*box_size )
        det_fm10 = det_fm10.view(n, -1, box_size)  # (N, 600, box_size), there are a total 600 boxes on this feature map
        
        det_fm5  = self.det_conv9_2(fm5)
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6*box_size )
        det_fm5 = det_fm5.view(n, -1, box_size)  # (N, 150, box_size), there are a total 150 boxes on this feature map
        
        det_fm3 = self.det_conv10_2(fm3)
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4*box_size )
        det_fm3 = det_fm3.view(n, -1, box_size)  # (N, 36, box_size), there are a total 36 boxes on this feature map
        
        det_fm1 = self.det_conv11_2(fm1)
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

    
    def my_post_process_deprecated(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold):
        ''' This approach based on my intuition that the box's class label should be the argmax of the softmax output,
        with this approach, score_threshold is not actually used properly since max of the softmax output is usually > 0.3
        And of course, this doesn't work well as the model's output is more biased toward class background, bc neg_pos_ratio=3
        So in many cases, the score for backdground overwhelm other classes like for example: (0.55, 0.01, 0.45, 0.04, 0.0,...),
        The result is that the recall is very low, it can't detect all the objects, however, precision is quite high, 
        like probably about >95%. But still, APs and mAP is low in general.
        
        Params:
            predicted_offsets: predicted offsets w.r.t the 8732 prior boxes, (gcxgcy), a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            score_threshold: minimum threshold for a box to be considered a match for a certain class
            iou_threshold: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        Return: 
            detections: (boxes, labels, and scores), they are lists of N tensors
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
        
        # for each image in the batch
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            
            # filter out boxes that are not qualified, that were predicted as background or with low confidence score
            qualify_mask = (class_ids[i] != 0) & (class_scores[i] > score_threshold) # (8732)
            qualified_boxes = predicted_offsets[i][qualify_mask]  # (n_qualified_boxes, 4)
            qualified_boxes_class = class_ids[i][qualify_mask]    # (n_qualified_boxes)
            qualified_boxes_score = class_scores[i][qualify_mask] # (n_qualified_boxes)
            
            if len(qualified_boxes) > 0:
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
    
    
    def my_post_process(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold, top_k=-1):
        ''' The differences from the previous my_post_process are:
        1: score_threshold is used to determine whether a box's class is background or objects
             E.g: let's say score_threshold=0.75, then boxes that have score of background class > 0.75 are considered background
        2: and then if the box contains an object, the object labels will be the argmax of the softmax output, background excluded
        Result:
        # See precision recall curve for more information
        # 2 times faster than post_process_top_k since a lot of background boxes was filtered out by score_threshold
        '''
        boxes = list()
        labels = list()
        scores = list()
        N, n_priors = predicted_offsets.shape[0:2]
        
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
        
        obj_masks = (predicted_scores[:,:,0] < score_threshold) # (N,8732)
        
        # for each image in the batch
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            obj_mask = obj_masks[i] # (8732)
            
            if obj_mask.sum().item() > 0:
                # filter out boxes that are background
                obj_boxes = predicted_offsets[i][obj_mask]  # (n_obj_boxes, 4) # n_obj_boxes: number of boxes containing object
                obj_boxes_score, obj_boxes_class = predicted_scores[i,:,1:self.n_classes][obj_mask].max(dim=1) # (n_obj_boxes)
                obj_boxes_class += 1 #since we excluded background class, argmax is between 0-19, we need to add 1 -> 1-20

                # convert to xy coordinates format
                obj_boxes = cxcy_to_xy(gcxgcy_to_cxcy(obj_boxes, self.priors_cxcy[obj_mask])) # (n_qualified_boxes, 4)

                # Non-max suppression
                for class_i in obj_boxes_class.unique(sorted=False).tolist():
                    class_mask = (obj_boxes_class == class_i)
                    boxes_class_i = obj_boxes[class_mask]
                    boxes_score_class_i = obj_boxes_score[class_mask]
                    
                    final_box_ids = nms(boxes_class_i, boxes_score_class_i, iou_threshold)  # (n_final_boxes after suppresion,)
                    
                    boxes_i.extend(boxes_class_i[final_box_ids].tolist())
                    labels_i.extend([class_i]*len(final_box_ids))
                    scores_i.extend(boxes_score_class_i[final_box_ids].tolist())
        
            boxes.append(torch.FloatTensor(boxes_i).to(device))
            labels.append(torch.LongTensor(labels_i).to(device))
            scores.append(torch.FloatTensor(scores_i).to(device))
            
            # Filter top k objects that have largest confidence score
            if boxes[i].size(0) > top_k and top_k > 0:
                scores[i], sort_ind = scores[i].sort(dim=0, descending=True)
                scores[i] = scores[i][:top_k]  # (top_k)
                boxes[i] = boxes[i][sort_ind[:top_k]]  # (top_k, 4)
                labels[i] = labels[i][sort_ind[:top_k]]  # (top_k)
        
        return boxes, labels, scores
    
    def post_process_top_k(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold, top_k):
        ''' return top_k detections sorted by confidence score
        Params:
            predicted_offsets: predicted offsets w.r.t the 8732 prior boxes, (gcxgcy), a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            score_threshold: minimum threshold for a box to be considered a match for a certain class
            iou_threshold: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            top_k: int, if the result contains more than k objects, just return k objects that have largest confidence score
        Return:
            detections: (boxes, labels, and scores), they are lists of N tensors
            boxes: N (n_boxes, 4)
            labels: N (n_boxes,)
            scores: N (n_boxes,)
        '''
        boxes = list()
        labels = list()
        scores = list()
        N, n_priors = predicted_offsets.shape[0:2]
        
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
        
        # for each image in the batch
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            
            # convert gcxgcy to xy coordinates format
            boxes_xy = cxcy_to_xy(gcxgcy_to_cxcy(predicted_offsets[i], self.priors_cxcy)) # (8732, 4)

            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                qualify_mask = class_scores > score_threshold
                n_qualified = qualify_mask.sum().item()
                if n_qualified == 0:
                    continue
                boxes_class_c = boxes_xy[qualify_mask]  # (n_qualified, 4)
                boxes_score_class_c = class_scores[qualify_mask]  # (n_qualified) <= 8732
                
                final_box_ids = nms(boxes_class_c, boxes_score_class_c, iou_threshold)  # (n_final_boxes,)
                
                boxes_i.extend(boxes_class_c[final_box_ids].tolist())
                labels_i.extend([c]*len(final_box_ids))
                scores_i.extend(boxes_score_class_c[final_box_ids].tolist())
        
            boxes.append(torch.FloatTensor(boxes_i).to(device))
            labels.append(torch.LongTensor(labels_i).to(device))
            scores.append(torch.FloatTensor(scores_i).to(device))
            
            # Filter top k objects that have largest confidence score
            if boxes[i].size(0) > top_k:
                scores[i], sort_ind = scores[i].sort(dim=0, descending=True)
                scores[i] = scores[i][:top_k]  # (top_k)
                boxes[i] = boxes[i][sort_ind[:top_k]]  # (top_k, 4)
                labels[i] = labels[i][sort_ind[:top_k]]  # (top_k)

        return boxes, labels, scores
    
    
    def inference(self, images, score_threshold, iou_threshold, top_k):
        ''' images: tensor size (N, 3, 300, 300), normalized
        '''
        predicted_offsets, predicted_scores = self.forward(images)
        return self.post_process_top_k(predicted_offsets, predicted_scores, score_threshold, iou_threshold, top_k)
        
    
if __name__ == "__main__":
    from loss import MultiBoxLoss
    torch.set_grad_enabled(False)
    
    MySSD300 = SSD300(n_classes = 21, vgg16_dir='models/')
    loss_func = MultiBoxLoss(priors_cxcy = MySSD300.priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.)
    
    #loss = loss_func.forward(predicted_offsets, predicted_scores, boxes, labels)
    #print(loss.item())

    # test detect objects
    #boxes, labels, scores = MySSD300.detect_objects(predicted_offsets, predicted_scores, score_threshold=0.6, iou_threshold=0.5)
    #breakpoint()
