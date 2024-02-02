import torch
import torch.nn.functional as F

from utils import box_ops

device = "cuda" if torch.cuda.is_available() else "cpu"

def bb_loss(bb_masks, bb_locpred, bb_locs, mask, args):
    batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
    
    # for mlp, output is batch_size*6 x dim, we reshape back to batch_size x 6 x dim
    
    batch_size = bb_locs.shape[0]

    
    for i in range(batch_size):
        pb, t = bb_locpred[i], bb_masks[i]
        mw, mh, target_bboxes = args.width, args.height, bb_locs[i]
        num_roles = mask[i].sum().detach().cpu().item()
        bbox_exist = bb_masks[i] != 0
        num_bbox = bbox_exist.sum().item()

        # bbox conf loss
        # loss_bbox_conf = F.binary_cross_entropy_with_logits(pbc[:num_roles], 
        #                                                     bbox_exist[:num_roles].float(), reduction='mean')
        # batch_bbox_conf_loss.append(loss_bbox_conf)

        
        # giou loss
        if num_bbox > 0: 
            loss_bbox = F.l1_loss(pb[bbox_exist], target_bboxes[bbox_exist], reduction='mean')
            batch_bbox_loss.append(loss_bbox.sum() / num_bbox)    
            #pb_og = box_ops.swig_box_cxcywh_to_xyxy(pb[bbox_exist], args.width, args.height, device=device)
            #tb_og =  box_ops.swig_box_cxcywh_to_xyxy(target_bboxes[bbox_exist], args.width, args.height, device=device)
            #loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(pb_og, tb_og))
            #batch_giou_loss.append(loss_giou.sum() / num_bbox)
            
    bbox_loss = torch.mean(torch.tensor(batch_bbox_loss)) #+ torch.mean(torch.tensor(batch_giou_loss)) 
        # + torch.mean(torch.tensor(batch_bbox_conf_loss))
    
    
    if bbox_loss == 0.0:
        bbox_loss = torch.tensor(0., device=device)
    
    return bbox_loss


            