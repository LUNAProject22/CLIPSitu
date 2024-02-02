
import torch
import torch.nn as nn
import torchvision as tv

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, encoder):
        super(MLP, self).__init__()
        self.num_ans_classes = encoder.get_num_labels()
        self.encoder = encoder
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.ReLU()
        ))
        
        for _ in range(num_layers - 1):
            block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.2),
                nn.ReLU()
            )
            self.blocks.append(block)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, self.num_ans_classes)
        )
    
    def forward(self, img_embeddings, verb_embeddings, role_embeddings,mask):
        feature_embeddings = torch.cat((img_embeddings.repeat_interleave(6,dim=0), verb_embeddings.repeat_interleave(6,dim=0), role_embeddings), 1)
        img_batch_size = img_embeddings.size(0)
        x = feature_embeddings
        out = x
        for i, block in enumerate(self.blocks):
            out = block(out)
        pred_embeddings = out
        logits = self.classifier(out)
        role_label_pred = logits.contiguous().view(img_batch_size, self.encoder.max_role_count, -1)
        return role_label_pred, pred_embeddings

def build_mlp_lnorm(input_size, num_layers, hidden_size, encoder):

    return MLP(input_size,num_layers, hidden_size, encoder)