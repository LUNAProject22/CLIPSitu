# %%
import torch
from PIL import Image
from collections import OrderedDict, defaultdict
import pickle, pickletools, gzip
import os
import numpy as np
import json 
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from transformers import CLIPModel, CLIPTokenizer, AutoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)


dataset_dir = '/data/dataset/imsitu'
anno_dir = os.path.join(dataset_dir, 'imSitu_annotations')
image_dir = os.path.join(dataset_dir, "of500_images")
space = json.load(open(os.path.join(anno_dir, "imsitu_space.json"),'r'))
noun_dict = space['nouns']

train_json = os.path.join(anno_dir, 'train.json')
val_json = os.path.join(anno_dir, 'dev.json')
test_json = os.path.join(anno_dir, 'test.json')

max_roles = 6 # based on imsitu characteristics

if os.path.exists('used_nouns.pkl'):
    all_nouns = pickle.load(open('used_nouns.pkl', 'rb'))
else:
    # for only used nouns in train + dev + test
    def get_used_nouns(json_file):
        label_dict = json.load(open(json_file,'r'))
        all_noun_classes = {}
        for imname in label_dict.keys():
            frames = label_dict[imname]['frames']
            roles = list(frames[0].keys())
            for role in roles:
                nouns = []
                for anno in range(2):
                    noun_class = frames[anno][role]
                    if noun_class:
                        all_noun_classes[noun_class] = 1
        all_nouns = {}
        for noun_class in all_noun_classes:
            nouns = noun_dict[noun_class]['gloss']
            if isinstance(nouns, list):
                noun = nouns[0]
            else:
                noun = nouns
            all_nouns[noun_class] = noun
        return all_nouns
        
    all_nouns = get_used_nouns(train_json)
    all_nouns.update(get_used_nouns(val_json))
    all_nouns.update(get_used_nouns(test_json))
    with open('used_nouns.pkl','wb') as f:
        pickle.dump(all_nouns, f, protocol=pickle.HIGHEST_PROTOCOL)

id2label = {i:v for i, (k,v) in enumerate(all_nouns.items())}
label2id = {v:i for i,v  in id2label.items()}
print("Total nouns: ", len(id2label.keys()))


class ImSituNounsDataset(object):
    
    def __init__(self, json_file):
        self.annotations = []
        self.verbs, self.nouns = space['verbs'], space['nouns']
                
        self.annotations = self._read_data(json_file)

        self.embeddings = {}
        for i in tqdm(range(len(self.annotations))):
            encoding = self.getitem(i)
            if encoding['image_emb'] is not None:
                self.embeddings[encoding['image_id']] = encoding
            # if (i+1)%100 == 0:
            #     break

    def __len__(self):
        return len(self.annotations)
    
    def _get_nouns(self):
        all_nouns = []
        for noun_class in self.nouns.keys():
            noun_words = self.nouns[noun_class]['gloss']
            all_nouns.extend(noun_words)
        return all_nouns
    
    def _read_data(self, json_file):
        annotations = []
        json_path = os.path.join(anno_dir, json_file)
        with open(json_path, "r") as f:
            label_dict = json.load(f)
            count = 0
            for imname in label_dict.keys():
                verb = label_dict[imname]['verb']
                subdir = imname.split('_')[0]
                impath = os.path.join(image_dir, subdir, imname)
                # print(impath)
                frames = label_dict[imname]['frames']
                roles = list(frames[0].keys())
                nouns = defaultdict(list)
                labels = defaultdict(list)
                roles_final = []
                for role in roles:
                    noun_classes = []
                    for anno in range(2):
                        noun_class = frames[anno][role]
                        if noun_class:
                            noun_classes.append(noun_class)
                    
                    
                    if noun_classes:
                        for noun_class in noun_classes:
                            noun = all_nouns[noun_class] # convert noun_class to noun
                            nouns[role].append(noun)
                            labels[role].append(label2id[noun]) # convert noun to label
                        roles_final.append(role)
                    
                annotation = {}
                annotation['image'] = impath
                annotation['verb'] = verb
                annotation['roles'] = roles_final
                annotation['labels'] = labels
                annotation['nouns'] = nouns
                annotations.append(annotation)
                
                count += 1
                # if count == 100:
                #     break
        return annotations
    
    def getitem(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        
        image = Image.open(annotation['image'])
        image = image.convert("RGB")
        
        encoding ={}
        try:
            image_input = image_processor(images=image, return_tensors="pt").to(device)
            outputs =  clip_model.vision_model.forward(**image_input)
            image_emb = clip_model.visual_projection(outputs['last_hidden_state']).detach().cpu()     
        except:
            # print('blank_image ', annotation['image'])
            encoding["image_id"] = annotation['image'].split('/')[-1]
            encoding['image_emb'] = None
            return encoding
        
        # text = []
        # for role in annotation['roles']:
        #     verb_role = annotation['verb'] + ' ' + role +'.'
        #     text.append(verb_role)

        # verb_inputs = clip_tokenizer(annotation['verb'], return_tensors="pt", padding=True).to(device)
        # verb_emb = clip_model.get_text_features(**verb_inputs).detach().cpu()
        
        # role_inputs = clip_tokenizer(annotation['roles'], return_tensors="pt", padding=True).to(device)
        # role_embs = clip_model.get_text_features(**role_inputs).detach().cpu()
        
        # verb_emb = verb_emb.repeat(role_embs.shape[0],1)
        # verb_role_embs = torch.cat((verb_emb, role_embs), dim=1)
        
        # value_embs = defaultdict()
        # for role in annotation['roles']:
        #     values = annotation['nouns'][role]
        #     value_inputs = clip_tokenizer(values, return_tensors="pt", padding=True).to(device)
        #     value_embs[role] = clip_model.get_text_features(**value_inputs).detach().cpu() # labels, dim

                
        role_mask = [True for _ in range(max_roles)]
        # if verb_role_embs.shape[0] < max_roles:
        #     num_roles = verb_role_embs.shape[0]
        #     to_pad = max_roles - num_roles
        #     # pad_tuple = last_dim_left, last_dim_right, second_last_dim_left, ...
        #     # verb_role_embs is num_roles, 7, 512
        #     verb_role_embs = F.pad(verb_role_embs, (0,0,0,to_pad), 'constant', 0) 
        num_roles = len(annotation['roles'])
        if num_roles < max_roles:                
            for i in range(num_roles, max_roles):
                role_mask[i] = False
        
        # expand to num_dimensions
        
        #role_mask = torch.BoolTensor(role_mask).unsqueeze(1)
        # role_mask = role_mask.repeat(1, verb_emb.shape[-1])
        #role_mask = role_mask.repeat(1, image_emb.shape[-1])    
            
        # encoding["verb_role_embs"] = verb_role_embs
        # encoding["verb"] = annotation['verb']
        # encoding["values"] = value_embs # dict of roles
        # encoding["roles"] = annotation['roles']
        # encoding["labels"] = annotation['labels']
        # encoding["role_mask"] = role_mask
        encoding[annotation['image'].split('/')[-1]] = image_emb

        return encoding

train_dataset = ImSituNounsDataset('train.json')
train_feat = train_dataset.embeddings
dev_dataset = ImSituNounsDataset('dev.json')
dev_feat = dev_dataset.embeddings
test_dataset = ImSituNounsDataset('test.json')
test_feat = test_dataset.embeddings
    
save_folder = './data/processed/imsitu_clip_xtf_features' #'/data/usrdata/roy/imsitu_clip_xtf_features'
with open(os.path.join(save_folder, 'clipfeats_multilabel_b32.pkl'), 'wb') as f:
    # pickled = pickle.dumps(overall_feat, protocol=pickle.HIGHEST_PROTOCOL)
    # optimized_pickle = pickletools.optimize(pickled)
    # f.write(optimized_pickle)
    pickle.dump(train_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dev_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
