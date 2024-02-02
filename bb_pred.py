import torch
import json
import os
import pickle
import random
import time
import argparse
import wandb

from collate import MyCollator, XTF_Collator

from utils.loss_functions import *
from torch import nn
from utils import utils, imsitu_loader, imsitu_encoder
from models import bb_models
from utils.misc import bb_intersection_over_union
from utils import box_ops

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, dev_loader, args, eval_frequency=2000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    # if model_saving_name == None:
    #     model_saving_name = model + '_' + datetime.now().strftime("%m%d-%H%M%S")
    #log_dir = 'runs/' + model_saving_name + datetime.now().strftime("%m%d-%H%M%S")
    #writer = SummaryWriter(log_dir=log_dir,filename_suffix='_'+model_name)
    model = model.to(device)
    bbox_loss = nn.L1Loss()
    max_score = 0. 
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        grnd_value, grnd_value_all = 0. , 0.
        total = 0.  
                     
        for i, (img_patch_embeddings, verb_embeddings, role_embeddings, label_embeddings, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(train_loader):
            #print('labels:', labels)
            #print('img"',img)
            #print('verb:',verb)
            #print(verb, verb.shape)
            total_steps += 1

            
            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_patch_embeddings = img_patch_embeddings.float().to(device)
            verb_embeddings = verb_embeddings.float().to(device)
            role_embeddings = role_embeddings.float().to(device)
            label_embeddings = label_embeddings.float().to(device)
            img_emb_verb_mlp = img_emb_verb_mlp.float().to(device)
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)

            if args.bb:
                bb_locpred  = model(img_patch_embeddings, verb_embeddings, role_embeddings, label_embeddings)
            
            bb_locpred = bb_locpred.reshape(-1,6,4)
            bbox_exist = bb_masks != 0
            loss = bbox_loss(bb_locpred[bbox_exist], bb_locs[bbox_exist])
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            args.optimizer.step()
            args.optimizer.zero_grad()

            train_loss += loss.item()
            
            
            batch_size = bb_locpred.shape[0]
            with torch.no_grad():
                for j in range(batch_size):
                    pb = box_ops.swig_box_cxcywh_to_xyxy(bb_locpred[j][bbox_exist[j]], 224, 224, device=device)
                    tb = box_ops.swig_box_cxcywh_to_xyxy(bb_locs[j][bbox_exist[j]], 224, 224, device=device)
                    num_bbox = tb.shape[0]
                    
                    bb_corrects = []
                    for k in range(num_bbox):
                        tbk = tb[k]
                        pbk = pb[k]
                        bb_correct = bb_intersection_over_union(pbk, tbk)
                        bb_corrects.append(bb_correct)
                        
                    total += 1
                    if sum(bb_corrects) == num_bbox:
                        grnd_value_all += 1
                    if sum(bb_corrects) == 1:
                        grnd_value += 1
                
                
            if total_steps % print_freq == 0:
                print ("{}, {}, {}, value:{}, value-all:{}, loss = {:.2f}, avg loss = {:.2}"
                    .format(total_steps-1, epoch, i, (grnd_value/total)*100, (grnd_value_all/total)*100, loss.item(),
                            train_loss / ((total_steps-1)%eval_frequency) ))

        epoch_time = time.time() - epoch_start_time
        value_dev, value_all_dev = eval(model, dev_loader, args)
        model.train()

       
        print ('Dev {} value:{} value-all:{}'.format(total_steps-1, value_dev*100, value_all_dev*100))
       

        if max_score < value_dev*100:
            torch.save(model.state_dict(), args.output_dir + "/{}".format(args.model_saving_name))
            max_score = value_dev*100
            print('Saving to:{}'.format(args.model_saving_name))
            print('New best model saved! {0}'.format(max_score))

        print('current train loss', train_loss)
        train_loss = 0
        
        print('Epoch ', epoch, ' completed in:',epoch_time,' seconds')
        args.scheduler.step()



def eval(model, loader, args, write_to_file = False):
    model.eval()
    grnd_value, grnd_value_all = 0. , 0.
    total = 0.                
    print ('evaluating model...')
    with torch.no_grad():

        for i, (img_patch_embeddings, verb_embeddings, role_embeddings, label_embeddings, imgs, verbs, labels, mask,\
            bb_masks, bb_locs, img_emb_verb_mlp) in enumerate(loader):
            
            verbs = verbs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            img_patch_embeddings = img_patch_embeddings.float().to(device)
            verb_embeddings = verb_embeddings.float().to(device)
            role_embeddings = role_embeddings.float().to(device)
            label_embeddings = label_embeddings.float().to(device)
            img_emb_verb_mlp = img_emb_verb_mlp.float().to(device)
            
            if args.bb:
                bb_masks = bb_masks.long().to(device)
                bb_locs =  bb_locs.float().to(device)

            if args.bb:
                bb_locpred  = model(img_patch_embeddings, verb_embeddings, role_embeddings, label_embeddings)
            
            bb_locpred = bb_locpred.reshape(-1,6,4)
            bbox_exist = bb_masks != 0                  
            batch_size = bb_locpred.shape[0]

            with torch.no_grad():
                for j in range(batch_size):
                    pb = box_ops.swig_box_cxcywh_to_xyxy(bb_locpred[j][bbox_exist[j]], 224, 224, device=device)
                    tb = box_ops.swig_box_cxcywh_to_xyxy(bb_locs[j][bbox_exist[j]], 224, 224, device=device)
                    num_bbox = tb.shape[0]
                    
                    bb_corrects = []
                    for k in range(num_bbox):
                        tbk = tb[k]
                        pbk = pb[k]
                        bb_correct = bb_intersection_over_union(pbk, tbk)
                        bb_corrects.append(bb_correct)
                        
                    total += 1
                    if sum(bb_corrects) == num_bbox:
                        grnd_value_all += 1
                    if sum(bb_corrects) == 1:
                        grnd_value += 1
            
                
    return grnd_value/total, grnd_value_all/total


def main():
    parser = argparse.ArgumentParser(description="imsitu verb predictor. Training, evaluation and prediction.")
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./bb_models', help='Location to output the model')
    parser.add_argument('--model', type=str, default='bb_mlp')
    parser.add_argument('--dataset_folder', type=str, default='./SWiG_jsons', help='Location of annotations')
    parser.add_argument('--train_file', default="train.json", type=str, help='trainfile name')       #train_freq2000.jaon
    parser.add_argument('--dev_file', default="dev.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test.json", type=str, help='test file name')
    parser.add_argument('--num_verb_classes', type=int, default=504)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--img_emb_base', default='vit-l14-336', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336'], help='xtf base')
    parser.add_argument('--num_tokens', type=int, default=50, help='for vit-b32')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_verb_layers', type=int, default=2)
    parser.add_argument('--pool', default='cls')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_scores', type=bool, default=False)
    parser.add_argument('--save_scores_dir', default='/data/output/saved_verb_preds')
    parser.add_argument('--img_emb_base_bb', default='vit-l14-336', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336'], help='xtf base')
    parser.add_argument('--bb', type=bool, default=True, help='predict bounding boxes')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()
    
    
    args.model_saving_name = "{}_catvr_lr{}_bs{}_nl{}_{}".format(args.model, args.lr, args.batch_size, args.num_verb_layers, args.img_emb_base_bb)

    
    constructor = 'build_%s' % args.model
    bb_model = getattr(bb_models, constructor)(args)
    print('Training {}'.format(args.model))
    
    args.optimizer = torch.optim.Adamax(bb_model.parameters(), lr=args.lr)    
    args.scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.9)
    
    #imgset_folder = '/data/dataset/imsitu/of500_images'
    imgset_folder = 'data/of500_images_resized'
    if args.img_emb_base_bb == 'vit-b16':
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-b16.pkl', 'rb'))
    elif args.img_emb_base_bb == 'vit-l14':
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-l14.pkl', 'rb'))
    elif args.img_emb_base_bb == 'vit-l14-336':
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-l14-336.pkl', 'rb'))
    else:
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-b32.pkl','rb'))
    args.img_emb_base = args.img_emb_base_bb
    text_dict = pickle.load(open('data/processed/clip_text_embeds.pkl','rb'))
    print('Loaded Embedding Dictionaries')

    train_set = json.load(open(args.dataset_folder + '/' + args.train_file))
    for missing in ['barbecuing_6.jpg', 'admiring_130.jpg','filming_66.jpg','kissing_171.jpg']: 
        train_set.pop(missing)

    encoder = imsitu_encoder.imsitu_encoder(train_set)
    args.encoder = encoder
    args.min_embed_loss = False

    train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, args.encoder,'train', args.encoder.train_transform)
    
    collate = XTF_Collator(img_dict, text_dict, args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,\
        collate_fn=collate, shuffle=True, num_workers=args.num_workers)

    dev_set = json.load(open(args.dataset_folder + '/' + args.dev_file))
    for missing in ['fueling_67.jpg', 'tripping_245.jpg']: 
        dev_set.pop(missing)
    dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size,\
        collate_fn=collate,shuffle=False, num_workers=args.num_workers)

    test_set = json.load(open(args.dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,\
        collate_fn=collate, shuffle=False, num_workers=0)

    if args.train:
        train(bb_model, train_loader, dev_loader, args)
    if args.eval:
        bb_model_weights = torch.load(args.load_model)
        bb_model.load_state_dict(bb_model_weights)
        grnd_val, grnd_val_all = eval(bb_model, dev_loader, args)
        print('Dev, grnd_value:{}, grnd_value-all:{}'.format(grnd_val, grnd_val_all))
        grnd_val, grnd_val_all = eval(bb_model, test_loader, args)
        print('Test, grnd_value:{}, grnd_value-all:{}'.format(grnd_val, grnd_val_all))

if __name__ == "__main__":
    main()

