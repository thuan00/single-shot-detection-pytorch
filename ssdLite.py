import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from utils import cxcy_to_xy, gcxgcy_to_cxcy, create_prior_boxes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, last_channel=1280, inverted_residual_setting=None, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.pretrained = pretrained
        block = InvertedResidual
        input_channel = 32

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.extras = nn.ModuleList([
            InvertedResidual(last_channel, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
        ])

        self.reset_parameters()
        if self.pretrained:
            pretrained_weights = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir='models/')
            self.load_state_dict(pretrained_weights, strict=False)
            

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        
        for i in range(14):
            x = self.features[i](x)
        features.append(x)  #fm20

        for i in range(14, len(self.features)):
            x = self.features[i](x)  #fm10
        features.append(x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)  #fm5, fm3

        return tuple(features)


class SSDLite(nn.Module):
    def __init__(self, n_classes):
        super(SSDLite, self).__init__()
        self.n_classes = n_classes
        self.backbone = MobileNetV2()
        
        # Detection layers 
        # Original SSDLite uses SeperableConvolution as detector layers, which is more efficient
        # However, I'll just gonna use basic Conv for simplicity and neglect speed performance
        self.det_fm20 = nn.Conv2d(96, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm10 = nn.Conv2d(1280, 6*(4+n_classes), kernel_size=3, padding=1)
        self.det_fm5 = nn.Conv2d(512, 6*(4+n_classes), kernel_size=3, padding=1)
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
        '''
        x: tensor (n, 3, 300, 300)
        '''
        n = x.size(0)
        
        # Feature extraction
        fm20, fm10, fm5, fm3 = self.backbone(x)

        # Detection
        box_size = 4 + self.n_classes  # each box has 25 values: 4 offset values and 21 class scores
        #
        det_fm20 = self.det_fm20(fm20)
        det_fm20 = det_fm20.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 2400, box_size)
        
        det_fm10 = self.det_fm10(fm10)
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 600, box_size)
        
        det_fm5 = self.det_fm5(fm5)
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 150, box_size)
        
        det_fm3 = self.det_fm3(fm3)
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 54, box_size)
        
        detection = torch.cat([det_fm20, det_fm10, det_fm5, det_fm3], dim=1)  # (N, 3204, box_size)
        offsets, class_scores = torch.split(detection, [4,self.n_classes], dim=2)
        
        return offsets, class_scores
    
    
    def get_prior_boxes(self):
        '''
        Return: 
            prior boxes in center-size coordinates, a tensor of dimensions (3204, 4)
        '''
        fmap_dims = {'fm20': 20,
                     'fm10': 10,
                     'fm5': 5,
                     'fm3': 3}

        obj_scales = {'fm20': 0.12,
                      'fm10': 0.24,
                      'fm5': 0.48,
                      'fm3': 0.768}

        aspect_ratios = {'fm20': [1., 2., 3., 0.5, .333],
                         'fm10': [1., 2., 3., 0.5, .333],
                         'fm5': [1., 2., 3., 0.5, .333],
                         'fm3': [1., 2., 3., 0.5, .333]}

        return create_prior_boxes(fmap_dims, obj_scales, aspect_ratios, last_scale=0.84)
    
    
    def post_process_top_k(self, predicted_offsets, predicted_scores, score_threshold, iou_threshold, top_k):
        ''' return top_k detections sorted by confidence score
        Params:
            predicted_offsets: predicted offsets w.r.t the 3206 prior boxes, (gcxgcy), a tensor of dimensions (N, 3204, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 3204, n_classes)
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
        
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 3204, n_classes)
        
        # for each image in the batch
        for i in range(N):
            boxes_i = list()
            labels_i = list()
            scores_i = list()
            
            # convert gcxgcy to xy coordinates format
            boxes_xy = cxcy_to_xy(gcxgcy_to_cxcy(predicted_offsets[i], self.priors_cxcy)) # (3204, 4)

            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (3204)
                qualify_mask = class_scores > score_threshold
                n_qualified = qualify_mask.sum().item()
                if n_qualified == 0:
                    continue
                boxes_class_c = boxes_xy[qualify_mask]  # (n_qualified, 4)
                boxes_score_class_c = class_scores[qualify_mask]  # (n_qualified) <= 3204
                
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