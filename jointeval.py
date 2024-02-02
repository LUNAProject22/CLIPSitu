import torch
from torch import nn
import torchvision
import numpy as np
import PIL
import os

from utils import utils, imsitu_scorer, imsitu_loader, imsitu_encoder


device = "cuda" if torch.cuda.is_available() else "cpu"

def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x

def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)

def get_topk_verbs(verb_logits, args, topk=1):
    sorted_idx = torch.sort(verb_logits, 1, True)[1]
    topk_predicted_verbs = sorted_idx[:,:topk]
    return topk_predicted_verbs


def get_role_embeds_from_verbs(verbs, args):
    """
    :param verbs: (batch_size)
    :return: (batch_size * max_role_count, 512)
    """
    role_embeddings = []
    for verb in verbs:
        verb_str = args.encoder.verb_list[verb]
        roles = args.encoder.verb2_role_dict[verb_str]
        for role_no in range(args.encoder.max_role_count):
            if role_no < len(roles):
                role = list(roles)[role_no]
                # role_features = args.verb_dict[verb_str][0][role]['role_features']
                role_features = args.text_dict[role].unsqueeze(0)
                role_embeddings.append(role_features)
            else:
                missing_role = torch.full((1, args.text_dim), -1)
                role_embeddings.append(missing_role)
    return torch.cat(role_embeddings, dim=0).float()



def jointeval(verb_mlp_model, role_model, dev_loader, args, write_to_file = False):
    verb_mlp_model.eval()
    role_model.eval()
    topk=5
    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        if args.verb_model == 'wiseft':
            transform = verb_mlp_model.val_preprocess
            if transform is not None:
                transform.transforms.insert(0, convert)
        for i, (img_embeddings, verb_embeddings_GT, role_embeddings_GT, label_embeddings, imgs, verbs, labels, mask, \
                bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(dev_loader):

            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            img_emb_verb_mlp = img_emb_verb_mlp.float().to(device)
            #centers = centers.float().to(device)

            if args.verb_model == 'wiseft':
                img_paths = ['data/of500_images_resized/' + img for img in imgs]
                img_list = []
                for image_name in img_paths:
                    data = transform(PIL.Image.open(image_name))
                    img_list.append(data)
                x = torch.stack(img_list).cuda()
                verb_predict = get_logits(x, verb_mlp_model)
                
            elif args.verb_model == 'verb_mlp':
                verb_predict = verb_mlp_model(img_emb_verb_mlp)        
            role_pred_topk = None
            sorted_idx = torch.sort(verb_predict, 1, True)[1]
            topk_predicted_verbs = sorted_idx[:,:topk]

            for k in range(0,topk):
                role_embeddings = []

                verbtopk = topk_predicted_verbs[:,k]
                verb_embeddings = [args.text_dict[args.encoder.verb_list[verb]] for verb in verbtopk]
   
                role_embeddings = get_role_embeds_from_verbs(verbtopk, args)

                if args.gpuid >= 0:
                    verb_embeddings = torch.stack(verb_embeddings).float().to(device)
                    role_embeddings = role_embeddings.float().to(device)

                # if args.learnable_verbs:
                #     # if learnable_verbs is True, return verb_list instead of verb_embeddings to model
                #     verb_embeddings = verbs
                if args.learnable_roles:
                    # if learnable_roles is True, return verb_list instead of role_embeddings to model
                    role_embeddings = verbs
                    
                if args.bb:
                    role_pred, pred_embed, bb_locpred  = role_model(img_embeddings, verb_embeddings, role_embeddings,mask)
                else:
                    role_pred, _  = role_model(img_embeddings, verb_embeddings, role_embeddings, mask)

                if args.bb:
                    bb_masks = bb_masks.long().to(device)
                    bb_locs =  bb_locs.float().to(device)
                    bb_locs = bb_locs[bb_masks]
                    bb_locpred = bb_locpred[bb_masks]

                if k == 0:
                    idx = torch.max(role_pred,-1)[1]
                    role_pred_topk = idx
                else:
                    idx = torch.max(role_pred,-1)[1]
                    role_pred_topk = torch.cat((role_pred_topk.clone(), idx), 1)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                del role_pred, idx, verbtopk, verb_embeddings, role_embeddings
            role_predict = role_pred_topk
            verb_predict = topk_predicted_verbs
            top1.add_point_eval5_log_sorted(imgs, verb_predict, verbs, role_predict, labels)
            top5.add_point_eval5_log_sorted(imgs, verb_predict, verbs, role_predict, labels)
            del verb_predict, imgs, verbs

    return top1, top5, 0

def jointeval_bb(verb_mlp_model, role_model, bb_model, dev_loader, args, write_to_file = False):
    verb_mlp_model.eval()
    role_model.eval()
    bb_model.eval()
    topk=5
    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        if args.verb_model == 'wiseft':
            transform = verb_mlp_model.val_preprocess
            if transform is not None:
                transform.transforms.insert(0, convert)
#         for i, (img_embeddings,verb_embeddings, role_embeddings, label_embeddings, imgs, verbs, labels, mask, label_strs, xtf_mask, centers) in enumerate(dev_loader):
        for i, (img_embeddings, verb_embeddings_GT, role_embeddings_GT, label_embeddings, imgs, verbs, labels, mask, \
                bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(dev_loader):

            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            img_emb_verb_mlp = img_emb_verb_mlp.float().to(device)
            #centers = centers.float().to(device)
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)

            # if args.verb_model == 'wiseft':
            #     img_paths = ['data/of500_images_resized/' + img for img in imgs]
            #     img_list = []
            #     for image_name in img_paths:
            #         data = transform(PIL.Image.open(image_name))
            #         img_list.append(data)
            #     x = torch.stack(img_list).cuda()
            #     verb_predict = get_logits(x, verb_mlp_model)
                
            # el
            if args.verb_model == 'verb_mlp':
                verb_predict = verb_mlp_model(img_emb_verb_mlp)        
            role_pred_topk = None
            sorted_idx = torch.sort(verb_predict, 1, True)[1]
            topk_predicted_verbs = sorted_idx[:,:topk]

            for k in range(0,topk):
                role_embeddings = []

                verbtopk = topk_predicted_verbs[:,k]
                verb_embeddings = [args.text_dict[args.encoder.verb_list[verb]] for verb in verbtopk]
                
                # role_embeddings = [verb_dict[encoder.verb_list[verb]][0][role]['role_features'] 
                #        for verb in verbtopk 
                #        for role in encoder.verb2_role_dict[encoder.verb_list[verb]]]
                # for verb in verbtopk:
                #     verb_str = args.encoder.verb_list[verb]
                #     roles = args.encoder.verb2_role_dict[verb_str]
                #     for role_no in range(max_roles):
                #         if(role_no < len(roles)):
                #             role = list(roles)[role_no]
                #             role_features = args.verb_dict[verb_str][0][role]['role_features']
                #             role_embeddings.append(role_features)
                #         else:
                #             missing_role = torch.full((1,512),-1)
                #             role_embeddings.append(missing_role)

                role_embeddings = get_role_embeds_from_verbs(verbtopk, args)

                if args.gpuid >= 0:
                    verb_embeddings = torch.stack(verb_embeddings).float().to(device)
                    role_embeddings = role_embeddings.float().to(device)

                role_pred, _, _ = role_model(img_embeddings, verb_embeddings, role_embeddings, mask)
                bb_locpred = bb_model(img_embeddings, verb_embeddings, role_embeddings, label_embeddings)
                if k == 0:
                    idx = torch.max(role_pred,-1)[1] #.transpose(0,1)
                    role_pred_topk = idx
                    bb_locpred_topk = [bb_locpred]
                else:
                    idx = torch.max(role_pred,-1)[1] #.transpose(0,1)
                    role_pred_topk = torch.cat((role_pred_topk.clone(), idx), 1)
                    bb_locpred_topk.append(bb_locpred)
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                del role_pred, idx, verbtopk, verb_embeddings, role_embeddings
            role_predict = role_pred_topk
            verb_predict = topk_predicted_verbs
            bb_predict = bb_locpred_topk
            top1.add_point_eval5_log_sorted_bb(imgs, verb_predict, verbs, role_predict, labels, bb_predict, bb_locs, bb_masks)
            top5.add_point_eval5_log_sorted_bb(imgs, verb_predict, verbs, role_predict, labels, bb_predict, bb_locs, bb_masks)

            del verb_predict, imgs, verbs

    return top1, top5, 0
