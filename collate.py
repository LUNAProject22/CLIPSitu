import torch
import json
import pickle
import random
import os
import csv
import numpy as np

global imsitu_space

#global xtf_feat_file

imsitu_space = json.load(open("imSitu/imsitu_space.json"))

def labels_to_terms(encoded_labels, encoder):
    term_memory = set()  # Keep track of terms we've already used
    nested_terms = []  # This will be the final result

    # Iterate over each list of labels for each annotator
    for labels in encoded_labels:
        terms = []  # This will hold the terms for the current annotator

        # Iterate over each label in the current list
        for label in labels:
            if label == len(encoder.label_list):  # If there's no role, add None and continue
                terms.append('')
                continue

            # Get the list of terms for the current label
            #term_list = id_to_terms.get(str(label), [])
            label_id = encoder.label_list[label]
            if label_id == '':  # If there's no role, add None and continue
                terms.append('')
                continue
            term_list = imsitu_space['nouns'][label_id]['gloss']

            # Find the first term that hasn't been used yet
            unused_term_found = False
            for term in term_list:
                if term not in term_memory:
                    terms.append(term)
                    term_memory.add(term)  # Remember that we've used this term
                    unused_term_found = True
                    break
            # If no unused term was found, add the first term in the list (if any)
            if not unused_term_found and term_list:
                terms.append(term_list[0])
            # else:  # If we didn't break from the loop, there were no unused terms
            #     terms.append(None)

        nested_terms.append(terms)

    return nested_terms

class MyCollator(object):
    """ Custom collator for the imsitu dataset. 
        Returns a batch of data in the form of a tuple of tensors.
    """
    def __init__(self, img_dict, text_dict, img_dict_verb, args):
        #self.dict = dict
        self.img_dict = img_dict
        self.img_dict_verb = img_dict_verb
        self.text_dict = text_dict
        #self.verb_dict = verb_dict
        #self.role_dict = role_dict
        #self.noun_dict = noun_dict
        self.encoder = args.encoder
        self.min_embed_loss = args.min_embed_loss

    def __call__(self, original_batch):
        img_embeddings = []
        img_emb_verb_mlp = []
        verb_embeddings = []
        role_embeddings = []
        label_embeddings = []
        batch_size = len(original_batch)
        verb_list = []
        labels_list = []
       # label_str_list = []
        img_list = []
        mask = []
        max_roles = self.encoder.max_role_count
        bb_locs = []
        bb_masks = []

        for img_id, verb, labels, bb_mask, bb_loc in original_batch:
            img_list.append(img_id)
            verb_list.append(verb)
            labels_list.append(labels)
            verb_str = self.encoder.verb_list[verb]
            roles = self.encoder.verb2_role_dict[verb_str]
            act_labels = labels_to_terms(labels.tolist(), self.encoder)
            if bb_loc is not None:
                bb_loc = list(bb_loc.detach())
            row_mask = []
            img_feature = self.img_dict[img_id]
            if len(img_feature.shape) == 1:
                img_feature = img_feature.unsqueeze(0)
            verb_feature = self.text_dict[verb_str].unsqueeze(0)
            emb_dim = verb_feature.shape[1]
            img_embeddings.append(img_feature)
            verb_embeddings.append(verb_feature)
            
            img_emb = self.img_dict_verb[img_id].unsqueeze(0)
            img_emb_verb_mlp.append(img_emb)
            
            for role_no in range(max_roles):
                if(role_no < len(roles)):
                    role = list(roles)[role_no]
                    role_features = self.text_dict[role].unsqueeze(0)
                    role_embeddings.append(role_features)
                    row_mask.append(1)
                    if self.min_embed_loss == True:
                        label_strs = [self.encoder.label_list[x[role_no]] for x in labels]
                        target_embedding = torch.cat([self.text_dict[x].unsqueeze(0) for x in label_strs], 0)
                    else:
                        annotator = random.randint(0, 2)
                        #label_str = act_labels[annotator][role_no]
                        label = self.encoder.label_list[labels[annotator][role_no]]
                        target_embedding = self.text_dict[label].unsqueeze(0)
                    label_embeddings.append(target_embedding)
                    #label_str_list.append(label_str)

                else:
                    row_mask.append(0)
                    if bb_loc != []:
                        if bb_mask is not None:
                            bb_mask.append(0)
                        if bb_loc is not None:
                            bb_loc.append(torch.tensor([-1,-1,-1,-1], dtype=torch.float32))
                    role_embeddings.append(torch.full((1,emb_dim),-1))
                    if self.min_embed_loss == True:
                        label_embeddings.append(torch.full((3,emb_dim),-1))
                    else:
                        label_embeddings.append(torch.full((1,emb_dim),-1))
                    #label_str_list.append('')
            mask.append(row_mask)
            if bb_loc != []:
                if bb_mask is not None:
                    bb_masks.append(bb_mask)
                if bb_loc is not None:
                    bb_locs.append(torch.stack(bb_loc))
        mask = torch.tensor(mask) 
        if bb_locs != []:
            if bb_masks != []:
                bb_masks = torch.tensor(bb_masks)
            bb_locs = torch.stack(bb_locs)
        #print('Batch size:{}, Feature embeddings size:{}, Labels size:{}'.format(batch_size,len(feature_embeddings),len(label_embeddings)))
        verb_list = torch.tensor(verb_list)
        labels_list = torch.stack(labels_list)
        return  torch.cat(img_embeddings), torch.cat(verb_embeddings),torch.cat(role_embeddings), torch.stack(label_embeddings).squeeze(), \
            img_list, verb_list, labels_list, mask, bb_masks, bb_locs, torch.cat(img_emb_verb_mlp) #, label_str_list


class XTF_Collator(object):
    """ Custom collator for the imsitu dataset for XTF Model. 
        Returns a batch of data in the form of a tuple of tensors.
    """
    def __init__(self, img_dict_verb, text_dict, args):
        #self.dict = dict
        self.img_dict_verb = img_dict_verb
        self.text_dict = text_dict
        self.encoder = args.encoder
        self.min_embed_loss = args.min_embed_loss
        xtf_base_folder = './data/processed/imsitu_clip_xtf_features'
        if args.img_emb_base == 'vit-b16':
            self.xtf_feat_file = pickle.load(open(os.path.join(xtf_base_folder,'xtf_clipfeats_b16.pkl'), 'rb'))
        elif args.img_emb_base == 'vit-l14':
            self.xtf_feat_file = pickle.load(open(os.path.join(xtf_base_folder,'xtf_clipfeats_l14.pkl'), 'rb'))
        elif args.img_emb_base == 'vit-l14-336':
            with open(os.path.join(xtf_base_folder,'xtf_clipfeats_l14_336.pkl'), 'rb') as f:
                self.xtf_feat_file = pickle.load(f)
                self.xtf_feat_file.update(pickle.load(f))
                self.xtf_feat_file.update(pickle.load(f))
        elif args.img_emb_base == 'align':
            #self.xtf_feat_file = pickle.load(open(os.path.join(xtf_base_folder,'xtf_alignfeats.pkl'), 'rb'))
            with open(os.path.join(xtf_base_folder,'xtf_alignfeats.pkl'), 'rb') as f:
                self.xtf_feat_file = pickle.load(f)
                self.xtf_feat_file.update(pickle.load(f))
                self.xtf_feat_file.update(pickle.load(f))
        else:
            self.xtf_feat_file = pickle.load(open(os.path.join(xtf_base_folder,'xtf_clipfeats_b32.pkl'), 'rb'))

        print('XTF feat file loaded successfully') 
        self.args = args
        #self.saved_centers = pickle.load(open('./data/processed/saved_centres.pkl', 'rb'))
        #print('loaded centres')

    def __call__(self, original_batch):
        xtf_img_embeddings = []
        img_emb_verb_mlp = []
        verb_embeddings = []
        role_embeddings = []
        label_embeddings = []
        batch_size = len(original_batch)
        verb_list = []
        labels_list = []
        #label_str_list = []
        img_list = []
        mask = []
        max_roles = self.encoder.max_role_count
    
        xtf_mask = []
        #centers = []
        bb_locs = []
        bb_masks = []
    
        for img_id, verb, labels, bb_mask, bb_loc in original_batch:
            img_list.append(img_id)
            verb_list.append(verb)
            labels_list.append(labels)
            verb_str = self.encoder.verb_list[verb]
            roles = self.encoder.verb2_role_dict[verb_str]
            act_labels = labels_to_terms(labels.tolist(), self.encoder)
            if bb_loc is not None:
                bb_loc = list(bb_loc.detach())
            # to take care of any images that are missing in the xtf_feat_file
            # after modularization does not serve any purpose as the missing images
            # have been removed while loading the jsons
            try:
                if self.args.img_emb_base == 'vit-l14-336':
                    img_feature = self.xtf_feat_file[img_id]['image_emb'][:,1:]
                else:
                    img_feature = self.xtf_feat_file[img_id][:,1:]
            except:
                xtf_mask.append(False)
                continue
            xtf_mask.append(True) 
            
            row_mask = []
            verb_feature = self.text_dict[verb_str].unsqueeze(0)
            emb_dim = verb_feature.shape[1]
            xtf_img_embeddings.append(img_feature)
            verb_embeddings.append(verb_feature)
            
            img_emb = self.img_dict_verb[img_id].unsqueeze(0)
            img_emb_verb_mlp.append(img_emb)
            #centers.append(self.saved_centers[verb_str])
            for role_no in range(max_roles):
                if(role_no < len(roles)):
                    role = list(roles)[role_no]
                    role_features = self.text_dict[role].unsqueeze(0)
                    role_embeddings.append(role_features)
                    row_mask.append(1)
                    if self.min_embed_loss == True:
                        label_strs = [self.encoder.label_list[x[role_no]] for x in labels]
                        target_embedding = torch.cat([self.text_dict[x].unsqueeze(0) for x in label_strs], 0)
                    else:
                        annotator = random.randint(0, 2)
                        #label_str = act_labels[annotator][role_no]
                        label = self.encoder.label_list[labels[annotator][role_no]]
                        target_embedding = self.text_dict[label].unsqueeze(0)
                    label_embeddings.append(target_embedding)

                else:
                    row_mask.append(0)
                    if bb_loc != []:
                        if bb_mask is not None:
                            bb_mask.append(0)
                        if bb_loc is not None:
                            bb_loc.append(torch.tensor([-1,-1,-1,-1], dtype=torch.float32))
                    role_embeddings.append(torch.full((1,emb_dim),-1))
                    if self.min_embed_loss == True:
                        label_embeddings.append(torch.full((3,emb_dim),-1))
                    else:
                        label_embeddings.append(torch.full((1,emb_dim),-1))
                    #label_str_list.append('')
            mask.append(row_mask)
            if bb_loc != []:
                if bb_mask is not None:
                    bb_masks.append(bb_mask)
                if bb_loc is not None:
                    bb_locs.append(torch.stack(bb_loc))
        mask = torch.tensor(mask) 
        if bb_locs != []:
            if bb_masks != []:
                bb_masks = torch.tensor(bb_masks)
            bb_locs = torch.stack(bb_locs)
        #print('Batch size:{}, Feature embeddings size:{}, Labels size:{}'.format(batch_size,len(feature_embeddings),len(label_embeddings)))
        verb_list = torch.tensor(verb_list)
        labels_list = torch.stack(labels_list)
        
        return torch.cat(xtf_img_embeddings),torch.cat(verb_embeddings),torch.cat(role_embeddings), torch.cat(label_embeddings),\
            img_list, verb_list, labels_list, mask, bb_masks, bb_locs, torch.cat(img_emb_verb_mlp)