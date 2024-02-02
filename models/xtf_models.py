import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce, repeat

from models.attention_blocks import CrossAttentionBlock, SelfAttentionBlock

device = "cuda" if torch.cuda.is_available() else "cpu"

    
    
class XTF(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.projection_layer = nn.Linear(args.text_dim, args.proj_dim)
        #self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        if args.sum:
            self.q_proj = nn.Linear(args.proj_dim, args.proj_dim)
        else:
            self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        # if args.img_emb_base == 'vit-l14' or args.img_emb_base == 'vit-l14-336':
        #     self.kv_proj = nn.Linear(768, args.proj_dim)
        # else:        
        #     self.kv_proj = nn.Linear(args.proj_dim, args.proj_dim)

        self.kv_proj = nn.Linear(args.image_dim, args.proj_dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers-1)])
        self.xatts = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers)])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads)
                
        self.output_proj = nn.Linear(args.proj_dim, args.proj_dim)
        self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))
        
        self.bbox_predictor = nn.Sequential(nn.Linear(args.proj_dim, args.proj_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(args.proj_dim*2, args.proj_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(args.proj_dim*2, 4))
        # self.bbox_conf_predictor = nn.Sequential(nn.Linear(args.proj_dim, args.proj_dim*2),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.2),
        #                                      nn.Linear(args.proj_dim*2, 1))
        
        self.args = args
        self.dim = args.proj_dim
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers=None):
    #def forward(self, q, kv, mask=None):
        #image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb
        verb_emb = self.projection_layer(verb_emb)
        role_emb = self.projection_layer(role_emb)
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)
        
        if self.args.sum:
            q = verb_emb_repeated + role_emb
        else:
            q = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        q = q.reshape(-1, 6, q.shape[1])
        kv = image_emb
        kv = self.kv_proj(kv)
        # convert mask from bs x num_roles to bs x num_roles x dim (default: 512)
        mask = mask.unsqueeze(-1).repeat(1,1,self.dim)
        q = self.q_proj(q)
        q = q + self.pos_emb
        
        for layer in self.xatts:
            q, kv, attn = layer(q, kv, mask)
        
        q = self.output_proj(q)
        logits = self.classifier(q)
        
        if self.args.bb:
            #bb_conf =  self.bbox_conf_predictor(q).sigmoid() # bs x num_roles 
            bb_locpred = self.bbox_predictor(q).sigmoid() # bs x num_roles x 4 
            return logits, q, bb_locpred
        else:
            return logits, q
        

class XTF_LearnTokens(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.projection_layer = nn.Linear(args.text_dim, args.proj_dim)
        #self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        if args.sum:
            self.q_proj = nn.Linear(args.proj_dim, args.proj_dim)
        else:
            self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        if args.img_emb_base == 'vit-l14' or args.img_emb_base == 'vit-l14-336':
            self.kv_proj = nn.Linear(768, args.proj_dim)
        else:        
            self.kv_proj = nn.Linear(args.proj_dim, args.proj_dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers-1)])
        self.xatts = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers)])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads)
                
        self.output_proj = nn.Linear(args.proj_dim, args.proj_dim)
        self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))

        self.num_roles = args.encoder.max_role_count
        verb_list = args.encoder.verb_list

        # Initialize verb tokens based on verb embeddings dictionary
        verb_tokens_temp = torch.zeros(len(verb_list), args.proj_dim)
        
        for verb_idx, verb in enumerate(verb_list):
            if verb in args.text_dict:
                verb_embeddings = args.text_dict[verb]
                verb_tokens_temp[verb_idx] = verb_embeddings
        self.verb_tokens = nn.Parameter(verb_tokens_temp)

        # Initialize role tokens based on role embeddings dictionary
        role_tokens_temp = torch.zeros(len(verb_list), self.num_roles, args.proj_dim)
        for verb_idx, verb in enumerate(verb_list):
            roles = args.encoder.verb2_role_dict[verb]
            for role_idx in range(self.num_roles):
                if(role_idx < len(roles)):
                    role_name = list(roles)[role_idx]
                    if role_name in args.text_dict:
                        role_embeddings = args.text_dict[role_name]
                        role_tokens_temp[verb_idx, role_idx] = role_embeddings
        self.role_tokens = nn.Parameter(role_tokens_temp)
                    
        # self.bbox_predictor = nn.Sequential(nn.Linear(args.proj_dim, args.proj_dim*2),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.2),
        #                                      nn.Linear(args.proj_dim*2, args.proj_dim*2),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.2),
        #                                      nn.Linear(args.proj_dim*2, 4))
        # self.bbox_conf_predictor = nn.Sequential(nn.Linear(args.proj_dim, args.proj_dim*2),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.2),
        #                                      nn.Linear(args.proj_dim*2, 1))


        # # mlp_inp = self.num_patches+args.proj_dim*2
        # # self.bbox_predictor = nn.ModuleList()
        # self.bbox_predictor.append(nn.Linear(mlp_inp, mlp_inp))
        # self.bbox_predictor.append(nn.Dropout(0.2))
        # self.bbox_predictor.append(nn.ReLU())
        # layer_norm = nn.LayerNorm(mlp_inp)
        # self.bbox_predictor.append(layer_norm)
        # self.bbox_predictor.append(nn.Linear(mlp_inp, 4))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(args.proj_dim, self.num_ans_classes)
        # )
        self.args = args
        self.dim = args.proj_dim
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers=None):
    #def forward(self, q, kv, mask=None):
        # image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb

        if self.args.learnable_verbs:
            # if verbs are learnt then batch of verbs is passed instead of verb_embeddings
            verbs = verb_emb
            verb_emb = torch.stack([self.verb_tokens[verb] for verb in verbs])
        verb_emb = self.projection_layer(verb_emb)
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)        

        if self.args.learnable_roles:
            # if roles are learnt, then batch of verbs is passed instead of role_embeddings
            verbs = role_emb
            batch_role_tokens = torch.stack([self.role_tokens[verb] for verb in verbs])
            role_emb = batch_role_tokens.reshape(role_emb.shape[0] * self.num_roles,-1)
        #role_emb = torch.rand([verb_emb.shape[0] * self.num_roles,512]).to(device)
        role_emb = self.projection_layer(role_emb)
        
        if self.args.sum:
            q = verb_emb_repeated + role_emb
        else:
            q = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        q = q.reshape(-1, 6, q.shape[1])
        kv = image_emb
        kv = self.kv_proj(kv)
        # convert mask from bs x num_roles to bs x num_roles x dim (default: 512)
        mask = mask.unsqueeze(-1).repeat(1,1,self.dim)
        q = self.q_proj(q)
        q = q + self.pos_emb
        
        #q, kv, attn = self.xatt(q,kv,mask)

        for layer in self.xatts:
            q, kv, attn = layer(q, kv, mask)
        
        q = self.output_proj(q)
        logits = self.classifier(q)
        
        if self.args.bb:
            #bb_conf =  self.bbox_conf_predictor(q).sigmoid() # bs x num_roles 
            #bb_locpred = self.bbox_predictor(q).sigmoid() # bs x num_roles x 4 
            bb_locpred = torch.cat([attn, verb_emb_repeated, role_emb], dim=-1)
            for layer in self.bbox_predictor:
                bb_locpred = layer(bb_locpred)

            return logits, q, bb_locpred
        else:
            return logits, q


class XTF_bb(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.projection_layer = nn.Linear(args.text_dim, args.proj_dim)
        if args.sum:
            self.q_proj = nn.Linear(args.proj_dim, args.proj_dim)
        else:
            self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        if args.img_emb_base == 'vit-l14' or args.img_emb_base == 'vit-l14-336':
            self.kv_proj = nn.Linear(768, args.proj_dim)
        else:        
            self.kv_proj = nn.Linear(args.proj_dim, args.proj_dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers-1)])
        self.xatts = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers-1)])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.num_heads)
                
        self.output_proj = nn.Linear(args.proj_dim, args.proj_dim)
        #self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))

        self.num_roles = args.encoder.max_role_count
                    
        if args.img_emb_base_bb == 'vit-b16':
            self.num_patches = 196 
        elif args.img_emb_base_bb == 'vit-l14':
            self.num_patches = 256
        elif args.img_emb_base_bb == 'vit-l14-336':
            self.num_patches = 576
        elif args.img_emb_base_bb == 'vit-b32':
            self.num_patches = 49

        mlp_inp = self.num_patches+args.proj_dim*2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(mlp_inp, mlp_inp))
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.ReLU())
        layer_norm = nn.LayerNorm(mlp_inp)
        self.layers.append(layer_norm)

        self.bbox_predictor = nn.Linear(mlp_inp, 4)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.proj_dim, self.num_ans_classes)
        )
        self.args = args
        self.dim = args.proj_dim
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers=None):
    #def forward(self, q, kv, mask=None):
        # image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb


        verb_emb = self.projection_layer(verb_emb)
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)        

        role_emb = self.projection_layer(role_emb)
        
        if self.args.sum:
            q = verb_emb_repeated + role_emb
        else:
            q = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        q = q.reshape(-1, 6, q.shape[1])
        kv = image_emb
        kv = self.kv_proj(kv)
        # convert mask from bs x num_roles to bs x num_roles x dim (default: 512)
        mask = mask.unsqueeze(-1).repeat(1,1,self.dim)
        q = self.q_proj(q)
        q = q + self.pos_emb
        
        q, kv, attn = self.xatt(q,kv,mask)
        for layer in self.xatts:
            q, kv, _ = layer(q, kv, mask)
        attn = attn.reshape(kv.shape[0]*6, -1)
        q = self.output_proj(q)
        logits = self.classifier(q)

        if self.args.bb:
            bb_locpred = torch.cat([attn, verb_emb_repeated, role_emb], dim=-1)
            for layer in self.layers:
                bb_locpred = layer(bb_locpred)
            bb_locpred = self.bbox_predictor(bb_locpred)
            # pred = self.output_proj(pred)
            #logits = self.classifier(pred)
            return logits, q, bb_locpred
        else:
            # pred = torch.cat([attn, verb_emb_repeated, role_emb], dim=-1)
            # for layer in self.layers:
            #     pred = layer(pred)
            # pred = self.classifier(pred)
            
#            logits = self.classifier(q)
            return logits, q


class XTFRole(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers-1)])
        #self.xatts = nn.ModuleList([CrossAttentionBlock(dim, max_roles, num_heads) for i in range(num_layers)-1])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads)
                
        self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))
        
        self.args = args
            
    def forward(self, image_emb, verb_emb, role_emb, mask, centers):
    #def forward(self, q, kv, mask=None):
        #image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb

        q = role_emb
        q = q.reshape(-1, 6, q.shape[-1])
        kv = image_emb
        # convert mask from bs x num_roles to bs x num_roles x 512
        mask = mask.unsqueeze(-1).repeat(1,1,512)
        

        q = q + self.pos_emb
        
        for layer in self.xatts:
            q, kv = layer(q, kv, mask)
                
        logits = self.classifier(q)
        
        return logits, q


class XTF7(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        
        self.xatts = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads) for i in range(args.num_layers)-1])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads)
                
        self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers):
    #def forward(self, q, kv, mask=None):
        #image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb
        role_emb = role_emb.view(-1, 6, role_emb.shape[-1])
        q = torch.cat((verb_emb.unsqueeze(1), role_emb), dim=1)
        
        kv = image_emb
        bs, num_roles, dim = role_emb.shape
        # convert mask from bs x num_roles to bs x num_roles x 512
        verb_mask = torch.ones(bs,1).bool().to(device)
        mask = torch.cat((verb_mask, mask), dim=1)
        mask = mask.unsqueeze(-1).repeat(1,1,512)
        
        q = q + self.pos_emb
        
        for layer in self.xatts:
            q, kv = layer(q, kv, mask)
                
        logits = self.classifier(q)
        logits_new = logits[:,1:].clone()
        q_new = q[:,1:].clone()
        
        return logits_new, q_new

   
class XTFVerbProto(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        
        # self.atts = nn.ModuleList([SelfAttentionBlock(dim, num_heads) for i in range(num_layers)])
        self.xatts = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads) for i in range(args.num_layers)])
        self.xatt_v = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads) for i in range(args.proto_layers)])
                
        self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))
        self.args = args
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers):
    #def forward(self, q, kv, mask=None):
        #image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb
        verb_q = verb_emb.repeat_interleave(self.args.encoder.max_role_count, dim=0).clone().detach()
        verb_q = verb_q.reshape(-1, self.args.encoder.max_role_count, verb_q.shape[-1])
        v_mask = torch.ones_like(verb_q).bool().to(device)
        verb_emb = verb_emb.unsqueeze(1)
        for l in self.xatt_v:
            verb_q, verb_emb = l(verb_q, verb_emb, v_mask)

        role_emb = role_emb.reshape(-1, self.args.encoder.max_role_count, role_emb.shape[-1])
        q = torch.cat([verb_q, role_emb], dim=-1)
        
        kv = image_emb
        # convert mask from bs x num_roles to bs x num_roles x 512
        mask = mask.unsqueeze(-1).repeat(1,1,512)
        q = self.q_proj(q)
        q = q + self.pos_emb
        
        for layer in self.xatts:
            q, kv = layer(q, kv, mask)
                
        logits = self.classifier(q)
        
        
        return logits, q



class XTFNoClassifier(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.q_proj = nn.Linear(args.proj_dim*2, args.proj_dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers)])
        # self.xatts = nn.ModuleList([CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads) for i in range(args.num_layers)])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads)
                
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))
        self.args = args
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers, gt_labels):
        #def forward(self, q, kv, mask=None):
        #image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)
        q = torch.cat([verb_emb_repeated, role_emb], dim=1)
        q = q.reshape(-1, 6, q.shape[1])
        kv = image_emb
        # convert mask from bs x num_roles to bs x num_roles x 512
        mask = mask.unsqueeze(-1).repeat(1,1,512)
        q = self.q_proj(q)
        q = q + self.pos_emb
        
        q, kv = self.xatt(q, kv, mask)
        for layer in self.atts:
            q = layer(q, mask)
        
        # q = self.output_proj(q)
                
        # Project the embeddings using the gt labels instead of classifier
        # q - bs*num_roles x 512
        # gt_labels - bs*num_roles x self.num_ans_classes
        # similarity - bs*num_roles x self.num_ans_classes x 512
        sim = gt_labels.view(-1,gt_labels.shape[-1]).unsqueeze(-1) @ q.view(-1, q.shape[-1]).unsqueeze(-2)
        # average along 512 dimension. reshape to original shape
        logits = torch.mean(sim, dim=-1).view(-1, self.args.encoder.max_role_count, self.num_ans_classes)

        return logits, q


class XTFLearnRole(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.projection_layer = nn.Linear(args.text_dim, args.dim)
        #self.q_proj = nn.Linear(args.dim*2, args.dim)
        self.role_tokens = nn.Parameter(args.encoder.max_role_count, args.dim)
        
        if args.sum:
            self.q_proj = nn.Linear(args.dim, args.dim)
        else:
            self.q_proj = nn.Linear(args.dim*2, args.dim)
        if args.img_emb_base == 'vit-l14' or args.img_emb_base == 'vit-l14-336':
            self.kv_proj = nn.Linear(768, args.dim)
        else:        
            self.kv_proj = nn.Linear(args.dim, args.dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.dim, args.num_heads) for i in range(args.num_layers-1)])
        self.xatts = nn.ModuleList([CrossAttentionBlock(args.dim, args.num_heads) for i in range(args.num_layers)])
        self.xatt = CrossAttentionBlock(args.dim, args.encoder.max_role_count, args.num_heads)
                
        self.output_proj = nn.Linear(args.dim, args.dim)
        self.classifier = nn.Linear(args.dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.dim))
        
        self.bbox_predictor = nn.Sequential(nn.Linear(args.dim, args.dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(args.dim*2, args.dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(args.dim*2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(args.dim, args.dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(args.dim*2, 1))
        
        self.args = args
        
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers=None):
    #def forward(self, q, kv, mask=None):
        #image_emb, verb_emb, role_emb,mask
        # mask is bs x num_roles x 512
        #q = verb_emb + role_emb
        #kv = image_emb
        verb_emb = self.projection_layer(verb_emb)
        role_emb = self.projection_layer(role_emb)
        verb_emb_repeated = verb_emb.repeat_interleave(6, dim=0)
        
        if self.args.sum:
            q = verb_emb_repeated + role_emb
        else:
            q = torch.cat([verb_emb_repeated, role_emb], dim=1)    
        q = q.reshape(-1, 6, q.shape[1])
        kv = image_emb
        kv = self.kv_proj(kv)
        # convert mask from bs x num_roles to bs x num_roles x 512
        mask = mask.unsqueeze(-1).repeat(1,1,512)
        q = self.q_proj(q)
        q = q + self.pos_emb
        
        for layer in self.xatts:
            q, kv = layer(q, kv, mask)
        
        q = self.output_proj(q)
        logits = self.classifier(q)
        
        if self.args.bb:
            bb_conf =  self.bbox_conf_predictor(q).sigmoid() # bs x num_roles 
            bb_locpred = self.bbox_predictor(q).sigmoid() # bs x num_roles x 4 
            return logits, q, bb_conf, bb_locpred
        else:
            return logits, q