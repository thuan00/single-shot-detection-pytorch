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
        
        # softmax
        #class_scores = F.softmax(class_scores, dim=2)  # (N, 8732, n_classes)
        
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
    
    
    def detect_objects_2(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).type_as(suppress))
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

        
    
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    MySSD300 = SSD300(n_classes = 21, vgg16_dir='models/')
    loss_func = MultiBoxLoss(priors_cxcy = MySSD300.priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.)
    
    # test loss function
    loss = loss_func.forward(predicted_offsets, predicted_scores, boxes, labels)
    print(loss.item())

    # test detect objects
    #boxes, labels, scores = MySSD300.detect_objects(predicted_offsets, predicted_scores, score_threshold=0.6, iou_threshold=0.5)
    #breakpoint()
    