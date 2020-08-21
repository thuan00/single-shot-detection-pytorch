import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import decimate, xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, find_jaccard_overlap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, x, y):
        ''' 
        x: (N, C)
        y: (N,)
        '''
        n_classes = x.size(1)
        
        # Convert y to one hot embedding 
        t = torch.eye(n_classes).to(device)[y] # (N,21)

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        w = w * (1-pt).pow(self.gamma)
        
        return F.binary_cross_entropy_with_logits(x, t, w.detach_(), reduction=self.reduction)


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., focal_loss=False):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        # loss functions
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='sum') if focal_loss else None
        
    def match_gt_priors(self, boxes, labels):
        ''' Given gt boxes, labels and (8732) priors, match them into the most suited priors
        N: batch size
        Params:
            boxes: true object bounding boxes in boundary coordinates, (xy), a list of N tensors: N(n_objects, 4)
            labels: true object labels, a list of N tensors: N(n_objects,)
        Return: 
            truth_offsets: tensor (N, 8732, 4)
            truth_classes: tensor (N, 8732,)
        '''
        N = len(boxes) #batch size
        n_priors = self.priors_cxcy.size(0)
        
        truth_offsets = torch.zeros((N, n_priors, 4), dtype=torch.float).to(device)
        truth_classes = torch.zeros((N, n_priors), dtype=torch.long).to(device)
        
        # for each image
        for i in range(N):
            n_objects = labels[i].shape[0]
            overlap = find_jaccard_overlap(self.priors_xy, boxes[i]) #(n_priors, n_boxes)
            
            # for each prior, find the max iou and the coresponding object id
            prior_iou, prior_obj = overlap.max(dim=1) #(n_priors)
            
            # for each object, find the most suited prior id
            _, object_prior = overlap.max(dim=0) #(n_objects)
            # for each object, assign its most suited prior with object id 
            for j in range(n_objects): prior_obj[object_prior[j]] = j
            # for each object, assign its most suited prior with hight iou to ensure it qualifies the thresholding 
            prior_iou[object_prior] = 1.
            
            # match bbox coordinates
            boxes_xy = boxes[i][prior_obj] # (8732, 4)
            
            # match prior class
            prior_class = labels[i][prior_obj]  # (8732)
            # thresholding: assign prior with iou < threshold to the class 0: background
            prior_class[prior_iou < self.threshold] = 0
            
            # save into the truth tensors
            truth_offsets[i,:,:] = cxcy_to_gcxgcy(xy_to_cxcy(boxes_xy), self.priors_cxcy)
            truth_classes[i,:] = prior_class
        
        return truth_offsets, truth_classes
    
    def forward(self, predicted_offsets, predicted_scores, boxes, labels):
        '''
        Params:
            predicted_offsets: predicted offsets w.r.t the 8732 prior boxes, (gcxgcy), a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            boxes: true  object bounding boxes in boundary coordinates, (xy), a list of N tensors: N(n_objects, 4)
            labels: true object labels, a list of N tensors: N(n_objects,)
        Return: 
            multibox loss, a scalar
        '''
        N = predicted_offsets.shape[0]
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        # Matching ground truth boxes (N, n_objects,4) to priors (N, 8732, 4)
        truth_offsets, truth_classes = self.match_gt_priors(boxes, labels)
        
        # Now we have ground truth priors and predicted priors we can now calculate the loss
        positive_priors = (truth_classes != 0) #(N,8732)
        n_positives = positive_priors.sum()  # (1)
        
        # Calculating loss = alpha*loc_loss + conf_loss
        # loc_loss: localization loss
        loc_loss = self.smooth_l1(predicted_offsets[positive_priors], truth_offsets[positive_priors])
        
        # Confidence loss
        if self.focal_loss is not None:
            conf_loss = self.focal_loss(predicted_scores.view(-1, n_classes), truth_classes.view(-1)) / n_positives
        else:# Hard negative mining
            full_conf_loss = self.cross_entropy(predicted_scores.view(-1, n_classes), truth_classes.view(-1)) #(N*n_priors)
            full_conf_loss = full_conf_loss.view(N, n_priors)
            # Since there is a huge unbalance between positive and negative priors so we only take the loss of the hard negatives

            n_hard_negatives = self.neg_pos_ratio * positive_priors.sum(dim=1)  # (N)
            conf_loss_hard_neg = 0
            # accummulate conf_loss_hard_neg for each sample in batch
            for i in range(N):
                conf_loss_neg,_ = full_conf_loss[i][~positive_priors[i]].sort(dim=0, descending=True)
                conf_loss_hard_neg = conf_loss_hard_neg + conf_loss_neg[0:n_hard_negatives[i]].sum()

            conf_loss = (full_conf_loss[positive_priors].sum() + conf_loss_hard_neg) / n_positives
        
        #print(loc_loss.item(), conf_loss.item())
        return self.alpha * loc_loss + conf_loss


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    predicted_offsets = torch.rand((1,3204,4))
    predicted_scores = torch.rand((1,3204,21))
    gt_boxes = [torch.Tensor([[0.1, 0.2, 0.3, 0.4]])]
    gt_labels = [torch.LongTensor([1])]
    
    from ssdLite import SSDLite
    model = SSDLite(n_classes=21)
    loss_func = MultiBoxLoss(priors_cxcy = model.get_prior_boxes(), threshold=0.5, neg_pos_ratio=3, alpha=1.)
    
    # test loss function
    loss = loss_func.forward(predicted_offsets, predicted_scores, gt_boxes, gt_labels)
    print(loss.item())
