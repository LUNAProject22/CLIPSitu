import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce, repeat

device = "cuda" if torch.cuda.is_available() else "cpu"
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

from torch import nn, einsum


class AbsPosEmb1DAISummer(nn.Module):
    """
    Given query q of shape [batch heads tokens dim] we multiply
    q by all the flattened absolute differences between tokens.
    Learned embedding representations are shared across heads
    """

    def __init__(self, tokens, dim_head):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: elements of the sequence
            dim_head: the size of the last dimension of q
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.abs_pos_emb = nn.Parameter(torch.randn(tokens, dim_head) * scale)

    def forward(self, q):
        return einsum('b h i d, j d -> b h i j', q, self.abs_pos_emb)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.drop_rate = drop
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(self.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y, mask=None):
              
        B1, N1, C = x.shape # bs, roles, dimension
        B2, N2, C = y.shape # bs, image_tokens, dimension
        assert B1 == B2, "role and image tokens should have same batch size"
        q = self.q(x).reshape(
            B1, N1, 1, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        
        kv = self.kv(y).reshape(
            B2, N2, 2, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Cross-Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C)

        x = self.proj(x)
        if mask is not None:
            if B1 > 1:
                x = mask*x
            else:
                x = mask.unsqueeze(1).float()*x            
        return x, y, attn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
    
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop_rate,
        )


    def forward(self, x, y, mask=None):
                        
        x_block, y, attn = self.attn(
                    self.norm1(x), 
                    self.norm2(y), 
                    mask=mask
                )
        x_norm = self.norm3(x_block)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, y, attn
    
# Self Attention

class SelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Joint space-time attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        if mask is not None:
            if B > 1:
                x = mask*x
            else:
                x = mask.unsqueeze(1).float()*x            
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop_rate,
        )
        
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop_rate,
        )


    def forward(self, x, mask):
        x_block = self.attn(self.norm1(x), mask)
        x_norm = self.norm2(x_block)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)
        return x
    
    
class XTF(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.num_ans_classes = args.encoder.get_num_labels()
        self.q_proj = nn.Linear(args.proj_dim, args.proj_dim)
        
        self.atts = nn.ModuleList([SelfAttentionBlock(args.proj_dim, args.num_heads) for i in range(args.num_layers-1)])
        #self.xatts = nn.ModuleList([CrossAttentionBlock(dim, max_roles, num_heads) for i in range(num_layers)-1])
        self.xatt = CrossAttentionBlock(args.proj_dim, args.encoder.max_role_count, args.num_heads)
                
        self.output_proj = nn.Linear(args.proj_dim, args.proj_dim)
        self.classifier = nn.Linear(args.proj_dim, self.num_ans_classes)
        
        self.pos_emb = torch.nn.Parameter(torch.randn(args.encoder.max_role_count, args.proj_dim))
    
    def forward(self, image_emb, verb_emb, role_emb, mask, centers=None):
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
        
        q = self.output_proj(q)
        logits = self.classifier(q)
        
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
