import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SSD300
from utils import decimate, xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, find_jaccard_overlap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def match_gt_priors(self, boxes, labels):
    ''' Given gt boxes, labels and (8732) priors, match them into the most suited priors 
    Params:
        boxes: true object bounding boxes in boundary coordinates, (xy), a list of N tensors: N(n_objects, 4)
        labels: true object labels, a list of N tensors: N(n_objects,)
    Return: 
        truth_offsets: tensor (N, 8732, 4)
        truth_classes: tensor (N, 8732,)
    '''
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
        n_positives = positive_priors.sum(dim=1)  # (N)
        
        # Calculating loss = alpha*loc_loss + conf_loss
        # loc_loss: localization loss
        loc_loss = self.smooth_l1(predicted_offsets[positive_priors], truth_offsets[positive_priors])
        
        # Confidence loss
        full_conf_loss = self.cross_entropy(predicted_scores.view(-1, n_classes), truth_classes.view(-1)) #(N*n_priors)
        full_conf_loss = full_conf_loss.view(N, n_priors)
        # Since there is a huge unbalance between positive and negative priors so we only take the loss of the hard negative priors

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
    
    loss_func = MultiBoxLoss(priors_cxcy = SSD300.get_priors_cxcy(), threshold=0.5, neg_pos_ratio=3, alpha=1.)
    
    # test loss function
    loss = loss_func.forward(predicted_offsets, predicted_scores, boxes, labels)
    print(loss.item())
