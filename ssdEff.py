import torch
import torch.nn as nn
import torch.nn.functional as F
from effnet.efficient_net_b3 import EfficientNet
from torchvision.ops import nms
from utils import cxcy_to_xy, gcxgcy_to_cxcy, create_prior_boxes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXTRAS = {
    'efficientnet-b3': [
        # in,  out, k, s, p
        [(384, 128, 1, 1, 0), (128, 256, 3, 2, 1)],  # 5 x 5
        [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 3 x 3
    ]
}

def add_extras(cfgs):
    extras = nn.ModuleList()
    for cfg in cfgs:
        extra = []
        for params in cfg:
            in_channels, out_channels, kernel_size, stride, padding = params
            extra.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            extra.append(nn.BatchNorm2d(in_channels))
            extra.append(nn.ReLU())
        extras.append(nn.Sequential(*extra))
    return extras


class SSDEff(nn.Module):
    def __init__(self, n_classes):
        super(SSDEff, self).__init__()
        self.n_classes = n_classes
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        ''' Efficient-net-b3 outputs
        fm38: N, 48, 38, 38
        fm19: N, 136, 19, 19
        fm10: N, 384, 10, 10
        '''
        self.extras = add_extras(EXTRAS['efficientnet-b3'])
        ''' extras conv outputs
        fm5: N, 256, 5, 5
        fm3: N, 256, 3, 3
        '''
        
        # FPN Lateral layers
        self.lat_fm19 = nn.Conv2d(136, 256, kernel_size=3, padding=1)
        self.lat_fm38 = nn.Conv2d(48, 256, kernel_size=3, padding=1)
        
        # FPN Top-down layers
        self.final_fm10 = nn.Conv2d(384, 256, kernel_size=1, padding=0)
        self.final_fm19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.final_fm38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Detection layer #shared across all feature maps
        #self.detection_layer = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm38 = nn.Conv2d(256, 4*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm19 = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm10 = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm5 = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm3 = nn.Conv2d(256, 6*(4+n_classes), kernel_size=3, padding=1)
        
        self.init_weights()
        self.priors_cxcy = self.get_prior_boxes()

    def init_weights(self):
        #Init weights for detection layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        ''' x: tensor (n, 3, 300, 300)
        '''
        n = x.size(0)
        
        # Bottom-up
        fm38, fm19, fm10 = self.backbone.extract_feature_maps(x)
        fm5 = self.extras[0](fm10)
        fm3 = self.extras[1](fm5)
        
        # Top-down + lateral connections
        fm10 = F.relu(self.final_fm10(fm10))
        
        fm19 = F.relu(self.lat_fm19(fm19)) + F.interpolate(fm10, size=(19,19), mode='nearest')
        fm19 = F.relu(self.final_fm19(fm19))
        
        fm38 = F.relu(self.lat_fm38(fm38)) + F.interpolate(fm19, size=(38,38), mode='nearest')
        fm38 = F.relu(self.final_fm38(fm38))

        # Detection
        box_size = 4 + self.n_classes  # each box has 25 values: 4 offset values and 21 class scores
        #
        det_fm38 = self.det_fm38(fm38)
        det_fm38 = det_fm38.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 8664, box_size)
        
        det_fm19 = self.det_fm19(fm19)
        det_fm19 = det_fm19.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 2166, box_size)
        
        det_fm10 = self.det_fm10(fm10)
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 600, box_size)
        
        det_fm5 = self.det_fm5(fm5)
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 150, box_size)
        
        det_fm3 = self.det_fm3(fm3)
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 54, box_size)
        
        detection = torch.cat([det_fm38, det_fm19, det_fm10, det_fm5, det_fm3], dim=1)  # (N, 11634, box_size)
        offsets, class_scores = torch.split(detection, [4,self.n_classes], dim=2)
        
        return offsets, class_scores
    
    
    def get_prior_boxes(self):
        '''
        Return: 
            prior boxes in center-size coordinates, a tensor of dimensions (11634, 4)
        '''
        fmap_dims = {'fm38': 38,
                     'fm19': 19,
                     'fm10': 10,
                     'fm5': 5,
                     'fm3': 3}

        obj_scales = {'fm38': 0.06,
                      'fm19': 0.12,
                      'fm10': 0.24,
                      'fm5': 0.48,
                      'fm3': 0.75}

        aspect_ratios = {'fm38': [1., 2., 3., 0.5, .333],
                         'fm19': [1., 2., 3., 0.5, .333],
                         'fm10': [1., 2., 3., 0.5, .333],
                         'fm5': [1., 2., 3., 0.5, .333],
                         'fm3': [1., 2., 3., 0.5, .333]}

        return create_prior_boxes(fmap_dims, obj_scales, aspect_ratios, last_scale=0.85)
    
    
    def post_process_top_k(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold, top_k):
        ''' return top_k detections sorted by confidence score
        Params:
            predicted_offsets: predicted offsets w.r.t the 3206 prior boxes, (gcxgcy), a tensor of dimensions (N, 11634, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 11634, n_classes)
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
        
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 11634, n_classes)
        
        # for each image in the batch
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            
            # convert gcxgcy to xy coordinates format
            boxes_xy = cxcy_to_xy(gcxgcy_to_cxcy(predicted_offsets[i], self.priors_cxcy)) # (11634, 4)

            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (11634)
                qualify_mask = class_scores > score_threshold
                n_qualified = qualify_mask.sum().item()
                if n_qualified == 0:
                    continue
                boxes_class_c = boxes_xy[qualify_mask]  # (n_qualified, 4)
                boxes_score_class_c = class_scores[qualify_mask]  # (n_qualified) <= 11634
                
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