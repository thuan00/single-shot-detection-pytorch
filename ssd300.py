import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from utils import decimate, cxcy_to_xy, gcxgcy_to_cxcy, create_prior_boxes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class VGGBackbone(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGGBackbone, self).__init__()
        self.features = features
        self.conv4_3_L2norm = L2Norm(512, scale=20)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features = []
        
        for i in range(23):
            x = self.features[i](x)
        x = self.conv4_3_L2norm(x)
        features.append(x)  #fm38

        for i in range(23, len(self.features)):
            x = self.features[i](x)
        features.append(x)  #fm19
        
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


model_url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
vgg16_backbone_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    # change pool4 ceil mode to True
    layers[16].ceil_mode = True
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers.extend([pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])
    
    return nn.Sequential(*layers)


def vgg16_backbone(cfg, pretrained, progress):
    model = VGGBackbone(make_layers(cfg, batch_norm=False))
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)
    return model


class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.backbone = vgg16_backbone(vgg16_backbone_cfg, pretrained=False, progress=False)
        
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
        
        # VGG16 Feature extraction
        fm38, fm19 = self.backbone(x)
        
        # Extra feature extraction
        x = F.relu(self.conv8_1(fm19))
        fm10 = F.relu(self.conv8_2(x))
        
        x = F.relu(self.conv9_1(fm10))
        fm5 = F.relu(self.conv9_2(x))
        
        x = F.relu(self.conv10_1(fm5))
        fm3 = F.relu(self.conv10_2(x))
        
        x = F.relu(self.conv11_1(fm3))
        fm1 = F.relu(self.conv11_2(x))

        # Detection
        box_size = 4 + self.n_classes  # each box has 25 values: 4 offset values and 21 class scores
        #
        det_fm38 = self.det_conv4_3(fm38)
        det_fm38 = det_fm38.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 5776, box_size)
        
        det_fm19 = self.det_conv7(fm19)
        det_fm19 = det_fm19.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 2166, box_size)
        
        det_fm10 = self.det_conv8_2(fm10)
        det_fm10 = det_fm10.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 600, box_size)
        
        det_fm5 = self.det_conv9_2(fm5)
        det_fm5 = det_fm5.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 150, box_size)
        
        det_fm3 = self.det_conv10_2(fm3)
        det_fm3 = det_fm3.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 36, box_size)
        
        det_fm1 = self.det_conv11_2(fm1)
        det_fm1 = det_fm1.permute(0, 2, 3, 1).contiguous().view(n, -1, box_size)  # (N, 4, box_size)
        
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

        obj_scales = {'conv4_3': 0.08,
                      'conv7': 0.16,
                      'conv8_2': 0.32,
                      'conv9_2': 0.54,
                      'conv10_2': 0.72,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
        
        return create_prior_boxes(fmap_dims, obj_scales, aspect_ratios, last_scale=1)

    
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
