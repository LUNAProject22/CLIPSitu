import torch

import torch.nn as nn
import random
import torch

import torch.nn as nn


# Normalization of a tensor
def normalize(A):
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0] 
    return A

# Exponential Cosine distance
def exp_cos(x1, x2): 
    sim_fun = nn.CosineSimilarity(dim=1, eps=1e-6) 
    dist = torch.exp(-sim_fun(x1,x2)) 
    return dist

# 1 - Cosine distance
def cos_dis(x1, x2): 
    sim_fun = nn.CosineSimilarity(dim=1, eps=1e-6) 
    dist = 1-sim_fun(x1,x2)
    return dist


# Jaccard vector similarity
def jvs(x1, x2, eps=1e-11):
    # x1 - bs x 512
    # x2 - bs x 512
    # mask - bs
    normalized_x1 = normalize(x1.clone().detach())
    normalized_x2 = normalize(x2.clone().detach())
    sim = torch.sum((2 * normalized_x1 * normalized_x2)/ (normalized_x1 * normalized_x1 + normalized_x2 * normalized_x2 + eps))
    return sim

# Jaccard vector similarity distance
def jvs_dis(x1, x2, eps=1e-11): 
    dis = -jvs(x1,x2,eps) 
    return dis

# similarity between role and noun using cross correlation
def ccrn(role_embs, pred_value_embs, mask):
    # role_embs contains role embedding for a verb - bs * 6 x 512
    # pred_value_embs contains pred_value_embs for a verb - bs * 6 x 512
    # mask - bs x 6 x 512
    
    sim_fun = nn.CosineSimilarity(dim=1, eps=1e-6)
    role_embs = role_embs.view(-1, 6, role_embs.shape[1])
    pred_value_embs = pred_value_embs.view(-1, 6, pred_value_embs.shape[1])
    bs = role_embs.shape[0]
    r_n_cor = []
    for b in range(bs):
        m = mask[b,:]
        r_n_cor.append(torch.sum(sim_fun(role_embs[b, m, :], pred_value_embs[b, m, :]))) # summation over roles
     
    return torch.tensor(r_n_cor)

    
# frobenius norm of difference between covariance of sim and 
def covdf(batch_target_value_embs, batch_pred_value_embs, mask):
    
    # batch_target_value_embs - bs*max_roles x 512
    # batch_pred_value_embs - bs*max_roles x 512
    # mask - bs*max_roles
    
    batch_target_value_embs_n = batch_target_value_embs[mask]
    batch_pred_value_embs_n = batch_pred_value_embs[mask]
    
    cov_target = torch.cov(normalize(batch_target_value_embs_n))
    cov_pred = torch.cov(normalize(batch_pred_value_embs_n))
    
    cov_diff_fro = torch.square(torch.linalg.matrix_norm(cov_target - cov_pred, ord='fro'))
    
    return cov_diff_fro


# Custom loss function for similarity
class CustomSimLoss(nn.Module):
    def __init__(self, sim):
        super(CustomSimLoss, self).__init__()
        self.sim = sim
        
    def forward(self, pred_emb, target_emb, mask):
        #sim = self.sim_loss(, targets[mask[:,0]])
        if self.sim == 'cos':
            #mask = mask.view(-1, mask.shape[-1])
            dis = cos_dis(pred_emb[mask], target_emb[mask])
            return torch.mean(dis)
        if self.sim == 'exp_cos':
            #mask = mask.view(-1, mask.shape[-1])
            cosine =  nn.CosineSimilarity(dim=1, eps=1e-6)
            dis = exp_cos(pred_emb[mask], target_emb[mask])
            return torch.mean(dis)
        if self.sim == 'jac':
            # mask = mask.view(-1, mask.shape[-1])[:,0]
            pred_emb = pred_emb[mask]
            target_emb = target_emb[mask]
            sim = jvs(pred_emb, target_emb)
            return 1-torch.mean(sim)
        if self.sim == 'ccrn':    
            sim = ccrn(pred_emb, target_emb, mask)
            return 1-torch.mean(sim)
        if self.sim == 'cov':
            sim = covdf(target_emb, pred_emb, mask)
            return torch.mean(torch.exp(-sim))

# Custom loss function for similarity that takes the minimum loss across annotators for cos and exp_cos
class CustomMinSimLoss(nn.Module):
    def __init__(self, sim):
        super(CustomMinSimLoss, self).__init__()
        self.sim = sim
        
    def forward(self, pred_emb, target_emb, mask):
        #sim = self.sim_loss(, targets[mask[:,0]])
        # Calculate the loss separately for each annotator
        loss_per_annotator = []
        for i in range(target_emb.size(1)):  # Iterate over the first dimension (annotators)
            target_emb_annotator = target_emb[:, i]
            if self.sim == 'cos':
                #mask = mask.view(-1, mask.shape[-1])
                dis = cos_dis(pred_emb[mask], target_emb_annotator[mask])
                loss_per_annotator.append(torch.mean(dis))

            if self.sim == 'exp_cos':
                #mask = mask.view(-1, mask.shape[-1])
                cosine =  nn.CosineSimilarity(dim=1, eps=1e-6)
                dis = exp_cos(pred_emb[mask], target_emb_annotator[mask])
                loss_per_annotator.append(torch.mean(dis))

        # Compute the minimum loss across annotators
        loss = torch.min(torch.stack(loss_per_annotator, dim=0))

        return loss


class ContrastiveLoss(nn.Module): 
    # Contrastive loss for similarity 
    def __init__(self, distance='cos', margin=0.01, num_negatives=1): 
        super(ContrastiveLoss, self).__init__() 
        self.eps = 1e-11 
        self.margin = margin 
        self.num_negatives = num_negatives
        if distance == 'cos': 
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cos_dis, margin=self.margin) 
        if distance == 'exp_cos': 
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=exp_cos, margin=self.margin) 
        if distance == 'jac': 
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=jvs_dis, margin=self.margin)
    
    def forward(self, pred_emb, target_emb, labels, mask): 
        # x1 - bs*6 x 512 
        # x2 - bs*6 x 512 
        # mask - bs 
        # labels - bs
        #mask = mask.view(-1, mask.shape[-1])[:,0]  # mask - bs x 6 x 512 -> bs*6 x 512 -> bs 
        pred_emb = pred_emb[mask] 
        target_emb = target_emb[mask,:] 
        anchors = pred_emb.clone().detach() 
        positives = target_emb.clone().detach() 
        masked_labels = [string for string, mask_value in zip(labels, mask.tolist()) if mask_value]

        negatives = [] 
        for i in range(len(masked_labels)): 
            candidates = [l for l in masked_labels if (l != masked_labels[i])] 
            for _ in range(self.num_negatives): 
                index = random.randint(0,len(candidates)-1) 
                negatives.append(target_emb[index,:]) 
        negatives = torch.stack(negatives) 
        positives = target_emb.repeat_interleave(self.num_negatives,dim=0) 
        anchors = pred_emb.repeat_interleave(self.num_negatives,dim=0) 
        output = self.triplet_loss(anchors, positives, negatives)
        return output



# class CustomTripletMarginLoss(nn.Module):
#     def __init__(self, distance_function, margin):
#         super(CustomTripletMarginLoss, self).__init__()
#         self.distance_function = distance_function
#         self.margin = margin

#     def forward(self, anchor, positive, negative, mask=None):
#         # Calculate the distance between anchor-positive and anchor-negative pairs using the distance_function and mask
#         ap_distance = self.distance_function(anchor, positive, mask)
#         an_distance = self.distance_function(anchor, negative, mask)

#         # Compute the triplet loss with the margin
#         loss = torch.relu(ap_distance - an_distance + self.margin)

#         # Return the mean loss over the batch
#         return torch.mean(loss)

# class ContrastiveLoss_OLD(nn.Module):
#     # Contrastive loss for similarity
#     def __init__(self,distance_function='cos', margin=0.01):
#         super(ContrastiveLoss_OLD, self).__init__()
#         self.eps = 1e-11
#         self.margin = margin
#         if distance_function=='jac':
#             self.distance_function = jvs
#         elif distance_function=='cos':
#             self.distance_function = exp_cos_sim


#     def forward(self, pred_emb, target_emb, label_strs, mask):
#         # x1 - bs x 512
#         # x2 - bs x 512
#         # mask - bs
#         negative_labels = []
#         for label in label_strs:
#             candidates = [l for l in label_strs if (l != label) and (l!='')]
#             negative_labels.append(random.choice(candidates))

#         # Convert negative_labels to a tensor of shape (batch_size,)
#         #negative_labels_tensor = torch.tensor(negative_labels)
#         negative_indices = []
#         for label in negative_labels:
#             negative_indices.append(label_strs.index(label))
#         label_embeddings = target_emb[mask]
#         anchors = label_embeddings.clone()
#         negatives = label_embeddings.clone()
#         empty_embedding = torch.full((512,), -1.0)
#         for i, index in enumerate(negative_indices):
#             if index >= 0:
#                 negatives[i] = label_embeddings[index]
#             else:
#                 negatives[i] = empty_embedding

#         triplet_loss = CustomTripletMarginLoss(distance_function=self.distance_function, margin=self.margin)
#         positives = pred_emb[mask]
#         output = triplet_loss(anchors, positives, negatives)

#         #sim = self.distance_function(pred_emb[mask[:,0]], target_emb[mask[:,0]], mask)
#         # sim = torch.sum ((2 * x1[mask,:] * x2[mask,:])/ (x1[mask,:] * x1[mask,:] + x2[mask,:] * x2[mask,:] + self.eps))
#         # sim = torch.mean(torch.exp(-sim))
#         return output


def cross_entropy_min_loss(role_label_pred, gt_labels, args):
    "Computes the cross entropy loss using the minimum loss across annotators"
    batch_size, max_roles, num_classes = role_label_pred.shape
    _, num_annotators, _ = gt_labels.shape
    criterion = nn.CrossEntropyLoss(ignore_index=args.encoder.get_num_labels(),reduction='none')

    # Expand prediction tensor to match gt_labels shape
    expanded_prediction = role_label_pred.unsqueeze(1).expand(-1, num_annotators, -1, -1)

    # Flatten the last two dimensions of gt_labels
    flattened_gt_labels = gt_labels.view(batch_size, num_annotators, -1)

    #Compute the cross-entropy loss for all annotators at once
    loss_all_annotators = criterion(expanded_prediction.reshape(-1, num_classes), flattened_gt_labels.reshape(-1))
    # Reshape the loss tensor to match the original dimensions
    reshaped_loss_all_annotators = loss_all_annotators.reshape(batch_size, num_annotators, max_roles)

    # Find the minimum loss across annotators
    min_loss, _ = torch.min(reshaped_loss_all_annotators, dim=1)

    # Compute the average loss across the batch
    batch_loss = torch.mean(min_loss)
    return batch_loss

def calculate_CE_loss(self, role_label_pred, gt_labels, args):
    batch_size = role_label_pred.size()[0]
    criterion = nn.CrossEntropyLoss(ignore_index=args.encoder.get_num_labels())
    gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* args.encoder.max_role_count*3, -1)

    role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
    role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
    role_label_pred = role_label_pred.transpose(0,1)
    role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))
    loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3
    return loss