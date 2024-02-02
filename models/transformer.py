import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class TransformerModel(nn.Module):
    def __init__(self, num_classes, num_layers, num_heads, args):
        super(TransformerModel, self).__init__()
        self.num_ans_classes = num_classes
        self.max_roles = args.encoder.max_role_count
        self.encoder = args.encoder
        self.image_emb = nn.Linear(args.image_dim, args.proj_dim)
        self.verb_emb = nn.Linear(args.text_dim, args.proj_dim)
        self.role_emb = nn.Linear(args.text_dim, args.proj_dim)
        self.input_layer = nn.Linear(args.proj_dim*3,args.proj_dim)
        self.args = args
        # Define the transformer layer
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=args.proj_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Linear(args.proj_dim, num_classes)
        self.bbox_predictor = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 4))
        

    def forward(self, image_emb, verb_emb, role_emb,mask):
        """Forward pass of the model"""
        # Encode the image and verb embeddings
        image_emb = self.image_emb(image_emb)
        verb_emb = self.verb_emb(verb_emb)
        role_emb = self.role_emb(role_emb)
        # Reshape the role embeddings
        batch_size = image_emb.size(0)
        #dim = role_emb.size(1)
        role_emb = role_emb.view(batch_size, -1, verb_emb.size(1))
         # Expand image and verb embeddings to account for 6 roles
        image_embeddings = image_emb.unsqueeze(1).repeat(1, self.max_roles, 1)
        verb_embeddings = verb_emb.unsqueeze(1).repeat(1, self.max_roles, 1)

        encodings = torch.cat((image_embeddings, verb_embeddings, role_emb), dim=2)
        encodings = encodings.permute(1, 0, 2)
        # Apply the transformer layer
        src = self.input_layer(encodings)
        mask = ~mask.bool() # invert the mask to match the transformer's masks
        output = self.transformer(src, src_key_padding_mask=mask)
        logits = self.classifier(output)
        logits = logits.permute(1, 0, 2)
        pred_embeddings = output
        role_label_pred = logits
        if self.args.bb:
            #bb_conf =  self.bbox_conf_predictor(pred_embeddings).sigmoid() # bs x num_roles 
            bb_locpred = self.bbox_predictor(pred_embeddings).sigmoid()
            #bb_conf = bb_conf.reshape(-1, 6, bb_conf.shape[-1])
            bb_locpred = bb_locpred.reshape(-1, 6, bb_locpred.shape[-1])
            return role_label_pred, pred_embeddings, bb_locpred
        else:
            return role_label_pred, pred_embeddings
        #logits = logits.view(-1, self.encoder.max_role_count, self.num_ans_classes)
        #return logits,output


def build_transformer(num_ans_classes, num_layers, num_heads, args):

    return TransformerModel(num_ans_classes,num_layers, num_heads, args)