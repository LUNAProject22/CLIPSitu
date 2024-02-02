'''
PyTorch implementation of MLP based SR : https://arxiv.org/abs/1708.04320
Variable Hidden Layer MLP implementation with Role Classifier
'''

import torch
import torch.nn as nn
import torchvision as tv

class MLP(nn.Module):
    def __init__(self, input_size,num_layers, hidden_size, args):
        super(MLP, self).__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.encoder = args.encoder
        #self.batch_size = img_batch_size
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.text_projection_layer = nn.Linear(args.text_dim, args.proj_dim)
        self.image_projection_layer = nn.Linear(args.image_dim, args.proj_dim)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_layer = nn.Linear(input_size, self.hidden_size)
        self.layers.append(input_layer)
        layer_norm = nn.LayerNorm(args.hidden_size)
        self.layers.append(layer_norm)
        if args.num_layers>1:
            for _ in range(args.num_layers-1):
                self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                self.layers.append(nn.Dropout(0.2))
                self.layers.append(nn.ReLU())
            self.layers.append(layer_norm)
        else:
            self.layers.append(nn.Dropout(0.2))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_size, args.proj_dim))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.proj_dim, self.num_ans_classes)
        )
        self.bbox_predictor = nn.Sequential(nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(512, 4))
        # self.bbox_conf_predictor = nn.Sequential(nn.Linear(512, 512),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.2),
        #                                      nn.Linear(512, 1))
        self.args = args

    def forward(self, img_embeddings, verb_embeddings, role_embeddings,mask):
        # Define the linear projection layer
        img_embeddings = self.image_projection_layer(img_embeddings)
        verb_embeddings = self.text_projection_layer(verb_embeddings)
        role_embeddings = self.text_projection_layer(role_embeddings)
        feature_embeddings = torch.cat((img_embeddings.repeat_interleave(6,dim=0), verb_embeddings.repeat_interleave(6,dim=0), role_embeddings), 1)
        img_batch_size = img_embeddings.size(0)
        x = feature_embeddings
        for layer in self.layers:
            x = layer(x)
        pred_embeddings = x
        logits = self.classifier(pred_embeddings)
        role_label_pred = logits.contiguous().view(img_batch_size, self.encoder.max_role_count, -1)
        if self.args.bb:
            #bb_conf =  self.bbox_conf_predictor(pred_embeddings).sigmoid() # bs x num_roles 
            bb_locpred = self.bbox_predictor(pred_embeddings).sigmoid()
            #bb_conf = bb_conf.reshape(-1, 6, bb_conf.shape[-1])
            bb_locpred = bb_locpred.reshape(-1, 6, bb_locpred.shape[-1])
            return role_label_pred, pred_embeddings, bb_locpred
        else:
            return role_label_pred, pred_embeddings
    

def build_mlp(input_size, num_layers, hidden_size, args):

    return MLP(input_size, num_layers, hidden_size, args)