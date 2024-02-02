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
from utils import utils, imsitu_loader, imsitu_encoder, model_complexity
from models import verb_models


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
    xe_loss = nn.CrossEntropyLoss()
    max_score = 0. 
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        correct1, correct5 = 0. , 0.
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
            
            if args.model == 'verb_mlp':
                verb_predict = model(img_emb_verb_mlp)
                #verb_predict = model(img_patch_embeddings)
            elif args.model == 'verb_ptf':
                verb_predict = model(img_patch_embeddings)
                    
            loss = xe_loss(verb_predict, verbs)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            args.optimizer.step()
            args.optimizer.zero_grad()

            train_loss += loss.item()
            with torch.no_grad():
                pred = torch.argmax(verb_predict, dim=1)
                correct1 += (pred == verbs).sum().item()
                total += verbs.shape[0]
                _, pred_top5 = torch.topk(verb_predict, k=5, dim=1)
                for v in verbs:
                    if v in pred_top5:
                        correct5 += 1
                
            if total_steps % print_freq == 0:
                print ("{}, {}, {}, top1:{}, top5:{}, loss = {:.2f}, avg loss = {:.2}"
                    .format(total_steps-1, epoch, i, correct1/total*100, correct5/total*100, loss.item(),
                            train_loss / ((total_steps-1)%eval_frequency) ))

        epoch_time = time.time() - epoch_start_time
        top1_dev, top5_dev = eval(model, dev_loader, args)
        model.train()

       
        print ('Dev {} {} {}'.format(total_steps-1, top1_dev*100, top5_dev*100))
       

        if max_score < top1_dev*100:
            torch.save(model.state_dict(), args.output_dir + "/{}".format(args.model_saving_name))
            max_score = top1_dev*100
            print('Saving to:{}'.format(args.model_saving_name))
            print('New best model saved! {0}'.format(max_score))

        print('current train loss', train_loss)
        train_loss = 0
        
        print('Epoch ', epoch, ' completed in:',epoch_time,' seconds')
        args.scheduler.step()



def eval(model, loader, args, write_to_file = False):
    model.eval()
    correct1, correct5 = 0. , 0.
    total = 0.                
    print ('evaluating model...')
    xe_loss = nn.CrossEntropyLoss()
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
            
            if args.model == 'verb_mlp':
                verb_predict = model(img_emb_verb_mlp)
                #verb_predict = model(img_patch_embeddings)
            elif args.model == 'verb_ptf':
                verb_predict = model(img_patch_embeddings)
            
            with torch.no_grad():
                pred = torch.argmax(verb_predict, dim=1)
                correct1 += (pred == verbs).sum().item()
                total += verbs.shape[0]
                _, pred_top5 = torch.topk(verb_predict, k=5, dim=1)
                for v in verbs:
                    if v in pred_top5:
                        correct5 += 1
                
    return correct1/total, correct5/total


def save_scores(model, loader, args, write_to_file = False):
    model.eval()
    model.to(device)
    correct1, correct5 = 0. , 0.
    total = 0.  
    print ('save verb scores...')
    verb_scores = {}
    with torch.no_grad():
        count = 0
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
            
            if args.model == 'verb_mlp':
                verb_predict = model(img_emb_verb_mlp)
            elif args.model == 'verb_ptf':
                verb_predict = model(img_patch_embeddings)
            
            with torch.no_grad():
                pred = torch.argmax(verb_predict, dim=1)
                correct1 += (pred == verbs).sum().item()
                total += verbs.shape[0]
                _, pred_top5 = torch.topk(verb_predict, k=5, dim=1)
                for v in verbs:
                    if v in pred_top5:
                        correct5 += 1
                        
            for j in range(verb_predict.shape[0]): 
                count += 1
                verb_scores[imgs[j]] = verb_predict[j].detach().cpu()
        
        print('{} examples'.format(str(count)))   
        print ('Top1:{}, Top5:{}'.format((correct1/total)*100, (correct5/total)*100))   
    return verb_scores



def main():
    parser = argparse.ArgumentParser(description="imsitu verb predictor. Training, evaluation and prediction.")
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./verb_models', help='Location to output the model')
    parser.add_argument('--model', type=str, default='verb_mlp')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--train_file', default="train.json", type=str, help='trainfile name')       #train_freq2000.jaon
    parser.add_argument('--dev_file', default="dev.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test.json", type=str, help='test file name')
    parser.add_argument('--num_verb_classes', type=int, default=504)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    #parser.add_argument('--xtf_base', default='vit-b32', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336'], help='xtf base')
    parser.add_argument('--num_tokens', type=int, default=50, help='for vit-b32')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_verb_layers', type=int, default=1)
    parser.add_argument('--pool', default='cls')
    parser.add_argument('--proj_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_scores', type=bool, default=False)
    parser.add_argument('--save_scores_dir', default='/data/output/saved_verb_preds')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--img_emb_base_verb', default='vit-b32', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336','align'], help='xtf base')
    parser.add_argument('--get_model_complexity', action='store_true',help='get_model_complexity')
    args = parser.parse_args()
    
    
    args.model_saving_name = "{}_lr{}_bs{}_nl{}_{}".format(args.model, args.lr, args.batch_size, args.num_verb_layers, args.img_emb_base_verb)

    if args.img_emb_base_verb == 'vit-b16':
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-b16.pkl', 'rb'))
        args.image_dim = 512
    elif args.img_emb_base_verb == 'vit-l14':
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-l14.pkl', 'rb'))
        args.image_dim = 768
    elif args.img_emb_base_verb == 'vit-l14-336':
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-l14-336.pkl', 'rb'))
        args.image_dim = 768
    elif args.img_emb_base_verb == 'align':
        img_dict = pickle.load(open('data/processed/ALIGN_img_embeds.pickle','rb'))
        text_dict = pickle.load(open('data/processed/ALIGN_text_embeds.pickle','rb'))
        args.image_dim = 640
        args.text_dim = 640
    else:
        img_dict = pickle.load(open('data/processed/clip_img_embeds_vit-b32.pkl','rb'))
        args.image_dim = 512
    if args.img_emb_base_verb != 'align':
        text_dict = pickle.load(open('data/processed/clip_text_embeds.pkl','rb'))
        args.text_dim = 512
    print('Loaded Embedding Dictionaries')

    constructor = 'build_%s' % args.model
    verb_model = getattr(verb_models, constructor)(args)
    print('Training {}'.format(args.model))
    
    args.optimizer = torch.optim.Adamax(verb_model.parameters(), lr=args.lr)    
    args.scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.9)
    
    imgset_folder = '/data/dataset/imsitu/of500_images'

    train_set = json.load(open(args.dataset_folder + '/' + args.train_file))
    for missing in ['barbecuing_6.jpg', 'admiring_130.jpg','filming_66.jpg','kissing_171.jpg']: 
        train_set.pop(missing)

    encoder = imsitu_encoder.imsitu_encoder(train_set)
    args.encoder = encoder
    args.min_embed_loss = False

    train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, args.encoder,'train', args.encoder.train_transform)
    
    img_dict_verb = img_dict
    if args.model == 'verb_mlp':
        collate = MyCollator(img_dict, text_dict, img_dict_verb, args)
    elif args.model == 'verb_ptf':
        collate = XTF_Collator(img_dict, text_dict, img_dict_verb, args)

    pytorch_total_params = sum(p.numel() for p in verb_model.parameters() if p.requires_grad)
    print('Number of Model parameters: {}'.format(pytorch_total_params))   
    if args.get_model_complexity:
        if args.load_model is not None:
            model_nm = 'trained_models/' + args.load_model
            utils.load_net(model_nm, [verb_model])
        inference_time, std_time = model_complexity.measure_inference_time_verb(verb_model,args, device=device, repetitions=300)
        flops, params = model_complexity.measure_flops_verb(verb_model,args, device=device)
        print(f'average time: {inference_time:.3f} ms')
        print(f'params: {params}')
        print(f'performance: {flops:.3f} GFLOPs')
        return
    

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

    if args.save_scores:
        model_weights = os.path.join(args.output_dir, args.load_model)
        utils.load_net(model_weights, [verb_model])
        
        dev_scores = save_scores(verb_model, dev_loader, args)
        with open(os.path.join(args.save_scores_dir, 'dev_scores.pkl'), 'wb') as f:
            pickle.dump(dev_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
        test_scores = save_scores(verb_model, test_loader, args)
        with open(os.path.join(args.save_scores_dir, 'test_scores.pkl'), 'wb') as f:
            pickle.dump(test_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        train(verb_model, train_loader, dev_loader, args)

if __name__ == "__main__":
    main()

