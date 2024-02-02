import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, num_classes, encoder, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.num_ans_classes = num_classes
        self.max_roles = encoder.max_role_count
        self.encoder = encoder
        # Define the embedding layers
        self.image_emb = nn.Linear(512, 512)
        self.verb_emb = nn.Linear(512, 512)
        self.role_emb = nn.Linear(512, 512)
        #self.input_layer = nn.Linear(1536,512)
        
        # Define the transformer layer
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=1536, nhead=num_heads, dim_feedforward=2048, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers
        )
        #self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6) # 1,2, 4 layers and heads
        self.ouput_layer = nn.Linear(1536,512)
        # Define the classification layer
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, image_emb, verb_emb, role_emb,mask):
        """Forward pass of the model"""
        # Encode the image and verb embeddings
        image_emb = self.image_emb(image_emb)
        verb_emb = self.verb_emb(verb_emb)
        role_emb = self.role_emb(role_emb)
        # Reshape the role embeddings
        batch_size = image_emb.size(0)
        role_emb = role_emb.view(batch_size, -1, 512)
        #labels = labels.view(batch_size, -1, 512)
         # Expand image and verb embeddings to account for 6 roles
        image_embeddings = image_emb.unsqueeze(1).repeat(1, self.max_roles, 1)
        verb_embeddings = verb_emb.unsqueeze(1).repeat(1, self.max_roles, 1)

        encodings = torch.cat((image_embeddings, verb_embeddings, role_emb), dim=2)
        encodings = encodings.permute(1, 0, 2)
        # Apply the transformer layer
        #src = self.input_layer(encodings)
        mask = ~mask.bool() # invert the mask to match the transformer's masks
        output = self.transformer(encodings, src_key_padding_mask=mask)
        output = self.ouput_layer(output)
        # Apply the classification layer
        logits = self.classifier(output)
        return logits,output


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


    def cos_embedding_loss(self, pred_noun_embedding,target_embedding): #TODO add mask to ignore padding
        """Compute the cosine embedding loss for the noun embeddings"""
        loss_fn = nn.CosineEmbeddingLoss()
        total_loss = 0
        pred_noun_embedding = pred_noun_embedding.view(-1, 512)    # (batch_size, 6, 512) -> (batch_size*6, 512)
        target = torch.ones(pred_noun_embedding.size()[0]).cuda()
        for i in range(pred_noun_embedding.size()[0]):
            if (target_embedding[i]==-1).all():
                continue
            loss = loss_fn(pred_noun_embedding[i],target_embedding[i], target[i])
            total_loss += loss
        return total_loss


    def cross_entropy_loss(self, gt_verbs, role_label_pred, gt_labels):
        """Compute the cross entropy loss for the role labels"""
        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.num_ans_classes)
        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))
        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3    # to account for 3 sets of GT labels
        return loss

def build_transformer_1536(num_ans_classes, encoder, num_layers, num_heads):

    return TransformerModel(num_ans_classes, encoder,num_layers, num_heads)