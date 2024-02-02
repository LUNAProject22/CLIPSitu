'''
PyTorch implementation of GGNN based SR : https://arxiv.org/abs/1708.04320
GGNN implementation adapted from https://github.com/chingyaoc/ggnn.pytorch
'''

import torch
import torch.nn as nn
import torchvision as tv

class MLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 512)

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            return output

        def calculate_loss(self,pred_noun_embedding,target_embedding):
            #mask = target_embedding==-1
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred_noun_embedding,target_embedding)
            return loss
            

        def calculate_loss_skip_missing_labels(self,pred_noun_embedding,target_embedding):
            #mask = target_embedding==-1
            loss_fn = nn.MSELoss()
            #loss = loss_fn(pred_noun_embeddings,target_embeddings)
            total_loss = 0
            for i in range(pred_noun_embedding.size()[0]):
                if (target_embedding[i]==-1).all():
                    continue
                loss = loss_fn(pred_noun_embedding[i],target_embedding[i])
                total_loss += loss
            return total_loss