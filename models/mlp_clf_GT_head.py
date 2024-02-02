'''
PyTorch implementation of MLP based SR : https://arxiv.org/abs/1708.04320
Variable Hidden Layer MLP implementation with Role Classifier
'''

import torch
import torch.nn as nn
import torchvision as tv

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size,encoder):
        super(MLP, self).__init__()
        self.num_ans_classes = encoder.get_num_labels()
        self.encoder = encoder
        #self.batch_size = img_batch_size
        self.input_size = input_size
        # self.hidden_size  = hidden_size
        # self.layers = nn.ModuleList()
        # input_layer = nn.Linear(input_size, self.hidden_size)
        # self.layers.append(input_layer)
        # layer_norm = nn.LayerNorm(hidden_size)
        # self.layers.append(layer_norm)
        # self.layers.append(nn.Linear(self.hidden_size, 512))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, self.num_ans_classes)
        )

    def forward(self, x):
        img_batch_size = int(x.size(0) / self.encoder.max_role_count)
        # for layer in self.layers:
        #     x = layer(x)
        # pred_embeddings = x
        logits = self.classifier(x)
        role_label_pred = logits.contiguous().view(img_batch_size, self.encoder.max_role_count, -1)
        return role_label_pred #, pred_embeddings

    def calculate_loss_skip_missing_labels(self,pred_noun_embedding,target_embedding):
        loss_fn = nn.MSELoss()
        total_loss = 0
        for i in range(pred_noun_embedding.size()[0]):
            if (target_embedding[i]==-1).all():
                continue
            loss = loss_fn(pred_noun_embedding[i],target_embedding[i])
            total_loss += loss
        return total_loss
    
    def calculate_cos_embedding_loss(self, pred_noun_embedding,target_embedding):
        loss_fn = nn.CosineEmbeddingLoss()
        total_loss = 0
        target = torch.ones(pred_noun_embedding.size()[0]).cuda()
        for i in range(pred_noun_embedding.size()[0]):
            if (target_embedding[i]==-1).all():
                continue
            loss = loss_fn(pred_noun_embedding[i],target_embedding[i], target[i])
            total_loss += loss
        return total_loss

    def calculate_CE_loss(self, gt_verbs, role_label_pred, gt_labels):
        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.num_ans_classes)
        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))
        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3
        return loss
    
    def cross_entropy_min_loss(self, gt_verbs, role_label_pred, gt_labels):
        "Computes the cross entropy loss using the minimum loss across annotators"
        batch_size, max_roles, num_classes = role_label_pred.shape
        _, num_annotators, _ = gt_labels.shape

        criterion = nn.CrossEntropyLoss(ignore_index=self.num_ans_classes,reduction='none')

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