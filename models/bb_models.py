'''
This is the full CNN classifier for verb or if any single role classification needed.
'''

import torch
import torch.nn as nn

import torchvision as tv
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class BBPredILMul(nn.Module):
    def __init__(self, args):
        super(BBPredILMul, self).__init__()
        if args.img_emb_base_bb == 'vit-b16' or args.img_emb_base_bb == 'vit-b32':
            self.img_patch_dim = 512    
        elif args.img_emb_base_bb == 'vit-l14' or args.img_emb_base_bb == 'vit-l14-336':
            self.img_patch_dim = 768
            
        self.vr_concat_dim = 1024
        self.vr_proj = nn.Linear(self.vr_concat_dim, self.img_patch_dim)
        self.img_proj = nn.Linear(self.img_patch_dim, self.img_patch_dim)
        if args.img_emb_base_bb == 'vit-b16':
            self.num_patches = 196 
        elif args.img_emb_base_bb == 'vit-l14':
            self.num_patches = 256
        elif args.img_emb_base_bb == 'vit-l14-336':
            self.num_patches = 576
        elif args.img_emb_base_bb == 'vit-b32':
            self.num_patches = 49
        mlp_inp = self.num_patches+args.dim*2
        self.bbox_predictor = nn.Sequential(nn.Linear(mlp_inp, mlp_inp),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(mlp_inp, mlp_inp),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(mlp_inp, 4))
        # self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(mlp_inp, mlp_inp))
        # self.layers.append(nn.Dropout(0.2))
        # self.layers.append(nn.ReLU())
        # layer_norm = nn.LayerNorm(mlp_inp)
        # self.layers.append(layer_norm)
        # self.layers.append(nn.Linear(mlp_inp, 4))
        
    
    def forward(self, img_embeddings, verb_emb, role_emb, noun_emb):
        
        img_embeddings = img_embeddings.repeat_interleave(6,dim=0)
        x = self.img_proj(img_embeddings)
        
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)
        y = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        y = self.vr_proj(y).unsqueeze(-1)
        
        op = x @ y
        op = torch.cat([op.squeeze(-1), verb_emb_repeated, role_emb], dim=1)
        
        # for layer in self.layers:
            # op = layer(op)
        op = self.bbox_predictor(op)
        return op
    

def build_bb_mlp(args):

    #covnet = vgg16_modified(num_ans_classes)
    mlp = BBPredILMul(args)

    return mlp

class BBPredILMul2(nn.Module):
    def __init__(self, args):
        super(BBPredILMul2, self).__init__()
        if args.img_emb_base_verb == 'vit-b16' or args.img_emb_base_verb == 'vit-b32':
            self.img_patch_dim = 512    
        elif args.img_emb_base_verb == 'vit-l14' or args.img_emb_base_verb == 'vit-l14-336':
            self.img_patch_dim = 768
            
        self.vr_concat_dim = 1024
        self.vr_proj = nn.Linear(self.vr_concat_dim, self.img_patch_dim)
        self.img_proj = nn.Linear(self.img_patch_dim, self.img_patch_dim)
        if args.img_emb_base == 'vit-b16':
            self.num_patches = 196 
        elif args.img_emb_base == 'vit-l14':
            self.num_patches = 256
        elif args.img_emb_base == 'vit-l14-336':
            self.num_patches = 576
        elif args.img_emb_base == 'vit-b32':
            self.num_patches = 49
        mlp_inp = self.img_patch_dim + self.img_patch_dim
        self.bbox_predictor = nn.Sequential(nn.Linear(mlp_inp, mlp_inp*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(mlp_inp*2, mlp_inp*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(mlp_inp*2, 4))
    
    def forward(self, img_embeddings, verb_emb, role_emb):
        
        img_embeddings = img_embeddings.repeat_interleave(6,dim=0)
        x = self.img_proj(img_embeddings)
        
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)
        y = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        y = self.vr_proj(y).unsqueeze(-1)
        
        att = x @ y # b*num_patches*img_patch_dim x b*img_patch_dim*1 = b*num_patches*1
        op = (x.permute(0,2,1) @ att).squeeze(-1) # b*img_patch_dim*num_patches x b*num_patches*1 = b*img_patch_dim*1
        #op = F.softmax(op, dim=1)
        op = torch.cat([op, y.squeeze(-1)], dim=-1)
        op = self.bbox_predictor(op)

        return op
    

def build_bb_mlp2(args):

    #covnet = vgg16_modified(num_ans_classes)
    mlp = BBPredILMul2(args)

    return mlp

class BBPredILMulAttNoun(nn.Module):
    def __init__(self, args):
        super(BBPredILMulAttNoun, self).__init__()
        if args.img_emb_base_verb == 'vit-b16' or args.img_emb_base_verb == 'vit-b32':
            self.img_patch_dim = 512    
        elif args.img_emb_base_verb == 'vit-l14' or args.img_emb_base_verb == 'vit-l14-336':
            self.img_patch_dim = 768
            
        self.vr_concat_dim = 1024
        self.noun_emb_dim = 512
        self.vr_proj = nn.Linear(self.vr_concat_dim, self.img_patch_dim)
        self.n_proj = nn.Linear(self.noun_emb_dim, self.img_patch_dim)
        self.img_proj = nn.Linear(self.img_patch_dim, self.img_patch_dim)
        if args.img_emb_base == 'vit-b16':
            self.num_patches = 196 
        elif args.img_emb_base == 'vit-l14':
            self.num_patches = 256
        elif args.img_emb_base == 'vit-l14-336':
            self.num_patches = 576
        elif args.img_emb_base == 'vit-b32':
            self.num_patches = 49
        mlp_inp = self.num_patches
        self.bbox_predictor = nn.Sequential(nn.Linear(mlp_inp, mlp_inp*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(mlp_inp*2, mlp_inp),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(mlp_inp, 4))
    
    def forward(self, img_embeddings, verb_emb, role_emb, noun_emb):
        
        img_embeddings = img_embeddings.repeat_interleave(6,dim=0)
        x = self.img_proj(img_embeddings)
        
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)
        y = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        y = self.vr_proj(y).unsqueeze(-1)
        z = self.n_proj(noun_emb).unsqueeze(-1)
        x1 = x @ y 
        x2 =  x @ z
        op = x1.squeeze(-1) + x2.squeeze(-1)
        op = self.bbox_predictor(op)

        return op
    
def build_bb_mlp3(args):

    #covnet = vgg16_modified(num_ans_classes)
    mlp = BBPredILMulAttNoun(args)

    return mlp