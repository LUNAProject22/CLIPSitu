import torch
import time
import os
import wandb
from utils.loss_functions import *
#from XTF.cross_attention import XTF, XTFRole, XTF7, XTFVerbProto
from torch import nn
import numpy as np
import torchvision
import PIL
from utils import utils, imsitu_scorer
#from torch.utils.tensorboard import SummaryWriter
from jointeval import get_role_embeds_from_verbs
from utils.loss_functions import cross_entropy_min_loss

from utils.embed_loss import get_embed_loss

from utils.bb_losses import bb_loss

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

def train(model, train_loader, dev_loader, args, eval_frequency=2000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []
    if args.gpuid >= 0 :
        if args.distributed:
            local_rank = int(os.environ['LOCAL_RANK'])
            pmodel = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
        else:
            ngpus = 2
            device_array = [i for i in range(0,ngpus)]
            pmodel = torch.nn.DataParallel(model, device_ids=device_array)
            pmodel = model.to(device)
    else:
        pmodel = model
    bbox_loss = nn.L1Loss()

    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        if args.distributed: train_loader.sampler.set_epoch(epoch)
        for i, (img_embeddings,verb_embeddings, role_embeddings, label_embeddings, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(train_loader):
            total_steps += 1
            
            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            verb_embeddings = verb_embeddings.float().to(device)
            role_embeddings = role_embeddings.float().to(device)
            label_embeddings = label_embeddings.float().to(device)
            img_emb_verb_mlp = img_emb_verb_mlp.float().to(device)
            
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)

            if args.learnable_verbs:
                verb_embeddings = verbs
                # if learnable_verbs is True, return verb_list instead of verb_embeddings to model
            if args.learnable_roles:
                # if learnable_roles is True, return verb_list instead of role_embeddings to model
                role_embeddings = verbs
            
            if args.bb:
                noun_predict, pred_embed, bb_locpred  = pmodel(img_embeddings, verb_embeddings, role_embeddings,mask)
            else:
                noun_predict, pred_embed  = pmodel(img_embeddings, verb_embeddings, role_embeddings, mask)
            
            cross_entropy_loss = cross_entropy_min_loss(noun_predict, labels, args)
            
            # mask is used to mask out the -1 labels in image order: (batch_size, 6)
            # target mask is used to mask out the -1 labels in role order: (batch_size * 6)
            #target_mask = ~(label_embeddings == -1).all(dim=1)
            target_mask = mask.view(-1).bool()
            pred_noun_embedding = pred_embed.view(-1, 512)
            
            embed_loss = get_embed_loss(pred_noun_embedding,label_embeddings,target_mask, mask, role_embeddings, args)

            loss = cross_entropy_loss + (args.lamda * embed_loss)      # lambda_val = [0.1,1,10] 
            
            if args.bb:
                #bbloss = bb_loss(bb_masks, bb_locpred, bb_locs, mask, args)
                bb_locpred = bb_locpred.reshape(-1,6,4)
                bbox_exist = bb_masks != 0
                bbloss = bbox_loss(bb_locpred[bbox_exist], bb_locs[bbox_exist])
                loss = loss + (args.lamda * bbloss)
                wandb.log({'epoch': epoch, 'Cross Entropy Loss': cross_entropy_loss,'Embedding Loss': embed_loss, \
                        'BB Loss': bbloss, 'Loss': loss})
            else:
                wandb.log({'epoch': epoch, 'Cross Entropy Loss': cross_entropy_loss,'Embedding Loss': embed_loss, 'Loss': loss})
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            args.optimizer.step()
            args.optimizer.zero_grad()

            train_loss += loss.item()

            if args.bb:
                top1.add_point_noun_bb(verbs, noun_predict, labels, mask, bb_locpred, bb_masks, bb_locs, args)
                top5.add_point_noun_bb(verbs, noun_predict, labels, mask, bb_locpred, bb_masks, bb_locs, args)
            else:            
                top1.add_point_noun(verbs, noun_predict, labels)
                top5.add_point_noun(verbs, noun_predict, labels)

            if total_steps % print_freq == 0:
                if args.bb:
                    top1_a = top1.get_average_results_bb()
                    top5_a = top5.get_average_results_bb()
                else:
                    top1_a = top1.get_average_results_nouns()
                    top5_a = top5.get_average_results_nouns()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2}"
                    .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.4f}", "1-"),
                            utils.format_dict(top5_a,"{:.4f}","5-"), loss.item(),
                            train_loss / ((total_steps-1)%eval_frequency) ))

        epoch_time = time.time() - epoch_start_time
        top1, top5, val_loss = eval(model, dev_loader, args)
        model.train()

        if args.bb:
            top1_avg = top1.get_average_results_bb()
            top5_avg = top5.get_average_results_bb()
            wandb.log({'value*': top5_avg["value*"]*100, 'value-all*': top5_avg["value-all*"]*100,\
                'grnd_value*': top5_avg["grnd_value*"]*100, 'grnd_value-all*': top5_avg["grnd_value-all*"]*100,\
                'epoch': epoch})
            avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                        top1_avg["grnd_value*"] + top1_avg["grnd_value-all*"] + \
                        top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"] + \
                        top5_avg["grnd_value*"] + top5_avg["grnd_value-all*"]
            avg_score /= 12


        else:
            top1_avg = top1.get_average_results_nouns()
            top5_avg = top5.get_average_results_nouns()

            wandb.log({'value*': top5_avg["value*"]*100, 'value-all*': top5_avg["value-all*"]*100,'epoch': epoch})


            avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                        top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
            avg_score /= 8

        print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                        utils.format_dict(top1_avg,'{:.4f}', '1-'),
                                                        utils.format_dict(top5_avg, '{:.4f}', '5-')))
        dev_score_list.append(avg_score)
        max_score = max(dev_score_list)

        if max_score == dev_score_list[-1]:
            torch.save(model.state_dict(), args.output_dir + "/{}".format(args.model_saving_name))
            print('Saving to:{}'.format(args.model_saving_name))
            print('New best model saved! {0}'.format(max_score))

        print('current train loss', train_loss)
        train_loss = 0
        top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3)
        top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)

        del noun_predict, loss, imgs, verbs, labels
        print('Epoch ', epoch, ' completed in:',epoch_time,' seconds')
        args.scheduler.step()



def eval(model, loader, args, write_to_file = False):
    model.eval()

    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)
    with torch.no_grad():

        for i, (img_embeddings,verb_embeddings, role_embeddings, label_embeddings, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(loader):
            
            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            verb_embeddings = verb_embeddings.float().to(device)
            role_embeddings = role_embeddings.float().to(device)
            label_embeddings = label_embeddings.float().to(device)
            img_emb_verb_mlp = img_emb_verb_mlp.float().to(device)
            
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)

            if args.learnable_verbs:
                verb_embeddings = verbs
                # if learnable_verbs is True, return verb_list instead of verb_embeddings to model
            if args.learnable_roles:
                # if learnable_roles is True, return verb_list instead of role_embeddings to model
                role_embeddings = verbs
                
            if args.bb:
                noun_predict, pred_embed, bb_locpred  = model(img_embeddings, verb_embeddings, role_embeddings,mask)
                bb_locpred = bb_locpred.reshape(-1,6,4)

            else:
                noun_predict, pred_embed  = model(img_embeddings, verb_embeddings, role_embeddings, mask)
            
 
 
            if args.bb:
                top1.add_point_noun_bb(verbs, noun_predict, labels, mask, bb_locpred, bb_masks, bb_locs, args)
                top5.add_point_noun_bb(verbs, noun_predict, labels, mask, bb_locpred, bb_masks, bb_locs, args)
                del noun_predict, imgs, verbs, labels, bb_locpred 
            else:            
                top1.add_point_noun(verbs, noun_predict, labels)
                top5.add_point_noun(verbs, noun_predict, labels)
                del noun_predict, imgs, verbs, labels
            

    return top1, top5, 0

def save_noun_predictions(model, loader, args, write_to_file = False):
    model.eval()

    print ('evaluating model and saving predictions')
    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3, write_to_file)
    with torch.no_grad():
        noun_preds = {}
        for i, (img_embeddings,verb_embeddings, role_embeddings, label_embeddings, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(loader):
            
            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            verb_embeddings = verb_embeddings.float().to(device)
            role_embeddings = role_embeddings.float().to(device)
            label_embeddings = label_embeddings.float().to(device)
            
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)

            if args.bb:
                noun_predict, pred_embed, bb_locpred  = model(img_embeddings, verb_embeddings, role_embeddings,mask)
                
            else:
                noun_predict, pred_embed  = model(img_embeddings, verb_embeddings, role_embeddings, mask)
                

            noun_predict = noun_predict.detach().cpu()
            mask = mask.detach().cpu()
            for i, img in enumerate(imgs):
                masked_pred = noun_predict[i][mask[i].bool()]
                pred_nouns_cls = torch.argmax(masked_pred, dim=1)            
                pred_nouns = [ args.encoder.label_list[n.item()] for n in pred_nouns_cls ]
                noun_preds[img] = pred_nouns
                
    return noun_preds



def save_both_predictions(model,verb_model, loader, args, write_to_file = False):
    model.eval()

    print ('evaluating model and saving predictions')
    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3, write_to_file)
    with torch.no_grad():
        preds = {}
        if args.verb_model == 'wiseft':
            transform = verb_model.val_preprocess
            if transform is not None:
                transform.transforms.insert(0, convert)
        for i, (img_embeddings,verb_embeddings_GT, role_embeddings_GT, label_embeddings_GT, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(loader):
            
            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)
                heights =  heights.float().to(device)
                widths = widths.float().to(device)

            if args.verb_model == 'wiseft':
                img_paths = ['data/of500_images_resized/' + img for img in imgs]
                img_list = []
                for image_name in img_paths:
                    data = transform(PIL.Image.open(image_name))
                    img_list.append(data)
                x = torch.stack(img_list).cuda()
                verb_predict = get_logits(x, verb_model)
                
            elif args.verb_model == 'verb_mlp':
                verb_predict = verb_model(img_embeddings)

            top_pred_verbs_idx = torch.argmax(verb_predict, dim=1)
            verb_embeddings = [args.text_dict[args.encoder.verb_list[verb]] for verb in top_pred_verbs_idx]
            verb_strs = [args.encoder.verb_list[top_pred_verb] for top_pred_verb in top_pred_verbs_idx]
            role_embeddings = get_role_embeds_from_verbs(verb_strs, args)

            if args.gpuid >= 0:
                verb_embeddings = torch.cat(verb_embeddings).float().to(device)
                role_embeddings = role_embeddings.float().to(device)
            
            if args.bb:
                noun_predict, pred_embed, bb_locpred  = model(img_embeddings, verb_embeddings, role_embeddings,mask)
                
            else:
                noun_predict, pred_embed  = model(img_embeddings, verb_embeddings, role_embeddings, mask)
                

            noun_predict = noun_predict.detach().cpu()
            mask = mask.detach().cpu()
            for i, img in enumerate(imgs):
                inner_dict = {}
                pred_verbs_cls = torch.argmax(verb_predict[i])
                pred_verb = args.encoder.verb_list[pred_verbs_cls]
                inner_dict['verb'] = pred_verb
                #verb_preds[img] = pred_verb

                pred_nouns_cls = torch.argmax(noun_predict[i][mask[i].bool()], dim=1)            
                pred_nouns = [args.encoder.label_list[n.item()] for n in pred_nouns_cls ]
                inner_dict['nouns'] = pred_nouns
                #noun_preds[img] = pred_nouns
                preds[img] = inner_dict
                
    return preds



def choose_tensor(gt_tensor, pred_tensor):
    """
    Randomly choose between gt_tensor and pred_tensor for each row in the batch
    """
    batch_size = gt_tensor.size(0)
    
    # Generate random probabilities for choosing tensor1 or tensor2
    choice_prob = torch.rand(batch_size)
    
    # Create a mask based on the random probabilities
    mask = (choice_prob < 0.5).to(device)
    # Reduce pred tensor to 1D by randomly choosing one element from each row
    random_indices = torch.randint(pred_tensor.size(1), (batch_size, 1)).to(device)
    random_pred_tensor = torch.gather(pred_tensor, 1, random_indices)
    # Choose from tensor1 or randomly from tensor2 based on the mask
    chosen_tensor = torch.where(mask.unsqueeze(1), gt_tensor, random_pred_tensor)
    return chosen_tensor


def train_with_topkverb(verb_model, role_model, train_loader, dev_loader, args, topk=1, eval_frequency=2000):
    role_model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []
    if args.gpuid >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]
        #pmodel = torch.nn.DataParallel(model, device_ids=device_array)
        pmodel = role_model.to(device)
    else:
        pmodel = role_model

    top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        for i, (img_embeddings,verb_embeddings_GT, role_embeddings_GT, label_embeddings, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(train_loader):
            total_steps += 1

            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_embeddings = img_embeddings.float().to(device)
            verb_embeddings_GT = verb_embeddings_GT.float().to(device)
            role_embeddings_GT = role_embeddings_GT.float().to(device)
            label_embeddings = label_embeddings.float().to(device)
            
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)
                heights =  heights.float().to(device)
                widths = widths.float().to(device)

            verb_logits = verb_model(img_embeddings)
            #topk_verbs = get_topk_verbs(verb_logits, args, topk=topk)
            sorted_idx = torch.sort(verb_logits, 1, True)[1]
            topk_predicted_verbs = sorted_idx[:,:topk]
            chosen_verbs = choose_tensor(verbs.unsqueeze(1), topk_predicted_verbs)

            verb_embeddings = torch.stack([args.text_dict[args.encoder.verb_list[verb]] for verb in chosen_verbs])
            
            role_embeddings = []
            for verb in chosen_verbs:
                verb_str = args.encoder.verb_list[verb]
                roles = args.encoder.verb2_role_dict[verb_str]
                for role_no in range(args.encoder.max_role_count):
                    if(role_no < len(roles)):
                        role = list(roles)[role_no]
                        role_features = args.text_dict[role].unsqueeze(0)
                       # role_features = args.verb_dict[verb_str][0][role]['role_features']
                        role_embeddings.append(role_features)
                    else:
                        missing_role = torch.full((1,512),-1)
                        role_embeddings.append(missing_role)

            # verb_role_embeds = get_verb_and_role_embeds(verb_logits, args, topk=topk)
            # k = random.choice(topk)
            # verb_embeddings,role_embeddings = verb_role_embeds[k]
            #verbs = torch.argmax(verb_pred, dim=1)
            #verb_embeddings = [args.verb_dict[args.encoder.verb_list[verb]] for verb in verbs]
            role_embeddings = torch.cat(role_embeddings).float().to(device)
            verb_embeddings = verb_embeddings.float().to(device)

            if args.bb:
                noun_predict, pred_embed, bb_locpred  = pmodel(img_embeddings, verb_embeddings, role_embeddings,mask)
            else:
                noun_predict, pred_embed  = pmodel(img_embeddings, verb_embeddings, role_embeddings, mask)
            
            cross_entropy_loss = cross_entropy_min_loss(noun_predict, labels,args)
            
            # mask is used to mask out the -1 labels in image order: (batch_size, 6)
            # target mask is used to mask out the -1 labels in role order: (batch_size * 6)
            #target_mask = ~(label_embeddings == -1).all(dim=1)
            target_mask = mask.view(-1).bool()
            pred_noun_embedding = pred_embed.view(-1, 512)
            
            embed_loss = get_embed_loss(pred_noun_embedding,label_embeddings,target_mask, mask, role_embeddings, args)

            loss = cross_entropy_loss + (args.lamda * embed_loss)      # lambda_val = [0.1,1,10] 
            
            if args.bb:
                bbloss = bb_loss(bb_masks, bb_locpred, bb_locs, heights, widths, mask)
                loss = loss + (args.lamda * bbloss)
                wandb.log({'epoch': epoch, 'Cross Entropy Loss': cross_entropy_loss,'Embedding Loss': embed_loss, \
                        'BB Loss': bbloss, 'Loss': loss})

            wandb.log({'epoch': epoch, 'Cross Entropy Loss': cross_entropy_loss,'Embedding Loss': embed_loss, 'Loss': loss})
            loss.backward()

            torch.nn.utils.clip_grad_norm_(role_model.parameters(), args.clip_norm)

            args.optimizer.step()
            args.optimizer.zero_grad()

            train_loss += loss.item()

            if args.bb:
                top1.add_point_noun_bb(verbs, noun_predict, labels, mask, bb_locpred, bb_masks, bb_locs, heights, widths)
                top5.add_point_noun_bb(verbs, noun_predict, labels, mask, bb_locpred, bb_masks, bb_locs, heights, widths)
            else:            
                top1.add_point_noun(verbs, noun_predict, labels)
                top5.add_point_noun(verbs, noun_predict, labels)

            if total_steps % print_freq == 0:
                if args.bb:
                    top1_a = top1.get_average_results_nouns_bb()
                    top5_a = top5.get_average_results_nouns_bb()
                else:
                    top1_a = top1.get_average_results_nouns()
                    top5_a = top5.get_average_results_nouns()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2}"
                    .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.4f}", "1-"),
                            utils.format_dict(top5_a,"{:.4f}","5-"), loss.item(),
                            train_loss / ((total_steps-1)%eval_frequency) ))

        epoch_time = time.time() - epoch_start_time
        top1, top5, val_loss = eval(role_model, dev_loader, args)
        role_model.train()

        if args.bb:
            top1_avg = top1.get_average_results_nouns_bb()
            top5_avg = top5.get_average_results_nouns_bb()
            wandb.log({'value*': top5_avg["value*"]*100, 'value-all*': top5_avg["value-all*"]*100,\
                'grnd_value*': top5_avg["grnd_value*"]*100, 'grnd_value-all*': top5_avg["grnd_value-all*"]*100,\
                'epoch': epoch})
            avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                        top1_avg["grnd_value*"] + top1_avg["grnd_value-all*"] + \
                        top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"] + \
                        top5_avg["grnd_value*"] + top5_avg["grnd_value-all*"]
            avg_score /= 12


        else:
            top1_avg = top1.get_average_results_nouns()
            top5_avg = top5.get_average_results_nouns()

            wandb.log({'value*': top5_avg["value*"]*100, 'value-all*': top5_avg["value-all*"]*100,'epoch': epoch})


            avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                        top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
            avg_score /= 8

        print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                        utils.format_dict(top1_avg,'{:.4f}', '1-'),
                                                        utils.format_dict(top5_avg, '{:.4f}', '5-')))
        dev_score_list.append(avg_score)
        max_score = max(dev_score_list)

        if max_score == dev_score_list[-1]:
            torch.save(role_model.state_dict(), args.output_dir + "/{}".format(args.model_saving_name))
            print('Saving to:{}'.format(args.model_saving_name))
            print('New best model saved! {0}'.format(max_score))

        print('current train loss', train_loss)
        train_loss = 0
        top1 = imsitu_scorer.imsitu_scorer(args.encoder, 1, 3)
        top5 = imsitu_scorer.imsitu_scorer(args.encoder, 5, 3)

        del noun_predict, loss, imgs, verbs, labels
        print('Epoch ', epoch, ' completed in:',epoch_time,' seconds')
        args.scheduler.step()