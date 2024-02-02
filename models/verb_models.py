'''
This is the full CNN classifier for verb or if any single role classification needed.
'''

import torch
import torch.nn as nn

import torchvision as tv
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class VerbMLP(nn.Module):
    def __init__(self, args):
        super(VerbMLP, self).__init__()
        self.num_ans_classes = args.num_verb_classes
        self.fcs = nn.ModuleList()
        self.relu = nn.ReLU()
        if args.img_emb_base_verb == 'vit-b16' or args.img_emb_base_verb == 'vit-b32':
            self.fcs.append(nn.Linear(512, args.proj_dim*2))    
        elif args.img_emb_base_verb == 'vit-l14' or args.img_emb_base_verb == 'vit-l14-336':
            self.fcs.append(nn.Linear(768, args.proj_dim*2))
        elif args.img_emb_base_verb == 'align':
            self.fcs.append(nn.Linear(640, args.proj_dim*2))
        for i in range(args.num_verb_layers-1):
            self.fcs.append(nn.Linear(args.proj_dim*2, args.proj_dim*2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.proj_dim*2, self.num_ans_classes)
        )

    def forward(self,x):
        for layer in self.fcs:
            x = self.relu(layer(x))
        x = self.classifier(x)

        return x
    

def build_verb_mlp(args):

    #covnet = vgg16_modified(num_ans_classes)
    mlp = VerbMLP(args)

    return mlp

class VerbPTF(nn.Module):
    def __init__(self, args ):
        super(VerbPTF, self).__init__()
        self.num_ans_classes = args.num_verb_classes
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=args.proj_dim, nhead=args.num_heads, dim_feedforward=2048, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=args.num_layers
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, args.num_tokens + 1, args.proj_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.proj_dim))
        
        self.pool = args.pool
        self.to_latent = nn.Identity()

        # Add rearrange
        self.rearrange1 = Rearrange('b s d -> s b d')
        self.rearrange2 = Rearrange('s b d -> b s d')
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.classifier = nn.Sequential(
            nn.Linear(args.proj_dim, args.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(args.proj_dim),
            nn.Linear(args.proj_dim, self.num_ans_classes)
        )
        

    def forward(self,x):
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.rearrange1(x)
        x = self.transformer(x)
        x = self.rearrange2(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        
        return self.classifier(x)
    
def build_verb_ptf(args):

    #covnet = vgg16_modified(num_ans_classes)
    mlp = VerbPTF(args)
    return mlp