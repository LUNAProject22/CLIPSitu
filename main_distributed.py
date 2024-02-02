import torch
import json
import os
import pickle
import random
import time
import argparse
import wandb
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collate import MyCollator, XTF_Collator
from train_and_eval import train, eval, save_noun_predictions, save_both_predictions, train_with_topkverb
from jointeval import jointeval
from utils.loss_functions import *
from torch import nn
from utils import utils, imsitu_loader, imsitu_encoder
from models import transformer, mlp, verb_models, res_mlp, mlp_lnorm
from models.xtf_models import XTF
from argparse import Namespace 
from wiseft import WiseFT

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument('--distributed',type=bool, default=False,help='use DDP')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [load_model]')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
#    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--imgset_dir', type=str, default='data/of500_images_resized', help='Location of original images')
    parser.add_argument('--train_file', default="train.json", type=str, help='trainfile name')       #train_freq2000.jaon
    parser.add_argument('--dev_file', default="dev.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test.json", type=str, help='test file name')
    parser.add_argument('--model_saving_name', type=str, default=None, help='saving name of the output model')
    parser.add_argument('--load_model', type=str, default=None, help='loading name for evaluation')
    parser.add_argument('--debug',type=bool, default=False,help='debug case')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_verb_layers', type=int, default=3)
    parser.add_argument('--num_verb_classes', type=int, default=504)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=8192)
    parser.add_argument('--model', type=str, default='transformer', help='can take mlp, transformer, xtf')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--lamda', type=float, default=1, help='lambda value for the embedding loss function')
    parser.add_argument('--embed_loss_fn', type=str, default=None, help='embedding loss function to be used')
    parser.add_argument('--num_neg_samples', type=int, default=1, help='number of negative samples to be used in the contrastive loss function')
    parser.add_argument('--verbeval', action='store_true')
    parser.add_argument('--verb_model', type=str, default='wiseft', help='can take verb_mlp or wiseft')
    parser.add_argument('--text_embeds', type=str, default='clip', help='can take clip, glove or bert')
    parser.add_argument('--proj_dim', type=int, default=512) # dimension all embeddings are projected to, within models (also model dim)
    parser.add_argument('--min_embed_loss', action='store_true')
    parser.add_argument('--bb', type=bool, default=False, help='predict bounding boxes')
    parser.add_argument('--save_noun_predictions', action='store_true')
    parser.add_argument('--img_emb_base', default='vit-b32', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336','align'], help='pooled image features for MLP and TF')
    parser.add_argument('--img_emb_base_verb', default='vit-b32', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336'], help='pooled image features for verb MLP')
    parser.add_argument('--save_verb_and_noun_predictions', action='store_true')
    parser.add_argument('--train_topkverb', action='store_true')
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--topkverb', type=int, default=3)
    #parser.add_argument('--img_emb_base', default='vit-b32', choices=['vit-b16', 'vit-b32', 'vit-l14', 'vit-l14-336'], help='unpooled image features for XTF')
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--sum', default=False, help='summation of verb and role query')
    
    args = parser.parse_args()
    if args.distributed:
        init_distributed()

    global imsitu_space
    n_epoch = args.epochs
    batch_size = args.batch_size
    n_worker = args.num_workers

    if args.embed_loss_fn == None:
        loss_fn_fname = 'XEonly'
    else: 
        removestring =" %:/,.\\[]<>*?"
        loss_fn_fname = ''.join([c for c in args.embed_loss_fn if c not in removestring])
        if args.min_embed_loss:
            loss_fn_fname += 'minEmbLoss'
        if args.bb:
            loss_fn_fname += '_bbloss'
        

    if args.model=='transformer' or args.model=='xtf':
        print("Using {} model".format(args.model))
        args.hidden_size = None
        if args.learning_rate==None:
            args.learning_rate = 0.001
        if args.model_saving_name == None:
            if args.model == 'xtf':
                args.model_saving_name = "{}_{}_nh{}_nl{}_{}".format(args.model, args.img_emb_base, args.num_heads, args.num_layers, loss_fn_fname)
            elif args.model == 'transformer':
                args.model_saving_name = "{}_{}_nh{}_nl{}_{}".format(args.model, args.img_emb_base, args.num_heads, args.num_layers, loss_fn_fname)
        else:
            args.model_saving_name = args.model_saving_name + "_{}_{}_{}".format(args.model, args.num_heads, args.num_layers)
    elif args.model=='mlp' or args.model=='res_mlp' or args.model=='mlp_lnorm':
        print("Using MLP model")
        args.num_heads = None
        if args.learning_rate==None:
            args.learning_rate = 0.0001
        if args.model_saving_name == None:
            args.model_saving_name = "{}_{}_hs{}_nl{}_{}".format(args.model, args.img_emb_base, args.hidden_size, args.num_layers, loss_fn_fname)
        else:
            args.model_saving_name = args.model_saving_name + "_{}_{}_{}".format(args.model, args.hidden_size, args.num_layers)
    # print('Saving to:{}'.format(args.model_saving_name))
    
    # if os.path.exists('./trained_models/' + "{}".format(model_saving_name)):
        if args.model_saving_name[-2:].isdigit():
            args.model_saving_name = args.model_saving_name[:-2] + str(int(args.model_saving_name[-2:])+1)
        else:
            args.model_saving_name = args.model_saving_name + "_01"
    
   
    config = {
        "text_embeds": args.text_embeds,
        "epochs": n_epoch,
        "learning_rate": args.learning_rate,
        "batch_size": batch_size,
        "model": args.model,
        "model_name": args.model_saving_name,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "lamda": args.lamda,
        "embed_loss_fn": args.embed_loss_fn,
    }


    if args.evaluate or args.test or args.verbeval or args.nolog:
        mode="disabled"
    else:
        mode="online"
        
    tagname = args.model
    if args.model == 'xtf':
        tagname += '_'+ args.img_emb_base
    if args.bb:
        tagname += '_bb'
    
    if args.sum:
        tagname += '_sum'
    
    tags = [tagname]
    tags.append(args.text_embeds)
    wandb.init(
    # set the wandb project where this run will be logged
        mode=mode,
        project="Clip_Situ",
        name=args.model_saving_name,
        sync_tensorboard=False,
        notes= "{} model: {} ".format(args.model.upper(),args.model_saving_name),
        tags=tags,
        config=config,
    )

    if args.bb:
        dataset_folder = 'SWiG_jsons'
    else:
        dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    imsitu_space = json.load(open("imSitu/imsitu_space.json"))

    def load_embedding(filename):
        if not os.path.exists(filename):
            raise ValueError(f"File {filename} not found!")
        with open(filename, 'rb') as file:
            return pickle.load(file)

    # Define a mapping from argument values to filenames
    img_emb_base_mapping = {
        'vit-b16': ('data/processed/clip_img_embeds_vit-b16.pkl', 512),
        'vit-b32': ('data/processed/clip_img_embeds_vit-b32.pkl', 512),
        'vit-l14': ('data/processed/clip_img_embeds_vit-l14.pkl', 768),
        'vit-l14-336': ('data/processed/clip_img_embeds_vit-l14-336.pkl', 768),
        'align': ('data/processed/ALIGN_img_embeds.pickle', 640)

    }

    text_embeds_mapping = {
        'align': ('data/processed/ALIGN_text_embeds.pickle', 640),
        'clip': ('data/processed/clip_text_embeds.pkl', 512),
        'glove': ('data/processed/glove_text_embeds.pkl', 300),
        'bert': ('data/processed/bert_text_embeds.pkl', 768)
    }

    # Load image embeddings
    img_file, args.image_dim = img_emb_base_mapping.get(args.img_emb_base, ('data/processed/clip_img_embeds_vit-b32.pkl', 512))
    img_dict = load_embedding(img_file)
    #img_dict = load_embedding(img_emb_base_mapping.get(args.img_emb_base, 'data/processed/clip_img_embeds_vit-b32.pkl'))
    img_file_verb, _ = img_emb_base_mapping.get(args.img_emb_base_verb, ('data/processed/clip_img_embeds_vit-b32.pkl', 512))
    img_dict_verb = load_embedding(img_file_verb)

    # Load text embeddings and set dimensions
    if args.text_embeds in text_embeds_mapping:
        text_file, text_dim = text_embeds_mapping[args.text_embeds]
        text_dict = load_embedding(text_file)
        args.text_dim = text_dim

    # Special case for 'align'
    if args.text_embeds == 'align' or args.img_emb_base == 'align':
        img_dict = load_embedding('data/processed/ALIGN_img_embeds.pickle')
        args.image_dim = 640

    print('Loaded Embedding Dictionaries')

    train_set = json.load(open(dataset_folder + '/' + args.train_file))
    for missing in ['barbecuing_6.jpg', 'admiring_130.jpg','filming_66.jpg','kissing_171.jpg']: 
        train_set.pop(missing)
    
    args.encoder = imsitu_encoder.imsitu_encoder(train_set)

    train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, args.encoder,'train', args.encoder.train_transform)

    constructor = 'build_%s' % args.model
    if args.model=='transformer':
        model = getattr(transformer, constructor)(args.encoder.get_num_labels(), args.num_layers, args.num_heads, args)
        collate = MyCollator(img_dict, text_dict, img_dict_verb, args)

    elif args.model=='mlp':
        if args.img_emb_base == 'vit-b16' or args.img_emb_base == 'vit-b32':
            model = getattr(mlp, constructor)(args.proj_dim*3, args.num_layers, args.hidden_size, args)
        elif args.img_emb_base == 'vit-l14' or args.img_emb_base == 'vit-l14-336':
            model = getattr(mlp, constructor)(args.proj_dim*2+768, args.num_layers, args.hidden_size, args)
        elif args.img_emb_base == 'align':
            model = getattr(mlp, constructor)(args.proj_dim*3, args.num_layers, args.hidden_size, args)  # check if correct
        collate = MyCollator(img_dict, text_dict, img_dict_verb, args)

    elif args.model=='xtf':
        model = XTF(args)
        collate = XTF_Collator(img_dict_verb, text_dict, args) # img_dict introduced for jointeval

    elif args.model=='res_mlp':
        model = getattr(res_mlp, constructor)(args.proj_dim*3, args.num_layers, args.hidden_size, args.encoder)
        collate = MyCollator(img_dict, text_dict, img_dict_verb, args)
    
    elif args.model=='mlp_lnorm':
        model = getattr(mlp_lnorm, constructor)(args.proj_dim*3, args.num_layers, args.hidden_size, args.encoder)
        collate = MyCollator(img_dict, text_dict, img_dict_verb, args)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Model parameters: {}'.format(pytorch_total_params))   

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    for missing in ['fueling_67.jpg', 'tripping_245.jpg']: 
        dev_set.pop(missing)

    dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, args.encoder, 'val', args.encoder.dev_transform)
    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, args.encoder, 'test', args.encoder.dev_transform)
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True)  
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, collate_fn=collate, num_workers=n_worker,pin_memory=True)                                   
        dev_sampler = DistributedSampler(dataset=dev_set,shuffle=False)
        dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, sampler=dev_sampler, collate_fn=collate, num_workers=n_worker, pin_memory=True)
        test_sampler = DistributedSampler(dataset=test_set,shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, collate_fn=collate, num_workers=n_worker,pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,collate_fn=collate,shuffle=True, num_workers=n_worker)
        dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, args.encoder, 'val', args.encoder.dev_transform)
        dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, collate_fn=collate,shuffle=False, num_workers=n_worker)
        test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, args.encoder, 'test', args.encoder.dev_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, collate_fn=collate,shuffle=False, num_workers=n_worker)


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.manual_seed(args.seed)
    if args.gpuid >= 0:
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    if args.resume_training:
        print('Resume training from: {}'.format(args.load_model))
        args.train_all = True
        if len(args.load_model) == 0:
            raise Exception('[pretrained module] not specified')
        model_nm = 'trained_models/' + args.load_model
        utils.load_net(model_nm, [model])
        args.model_saving_name = 'trained_models/resume_' + args.load_model
        args.optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
        
    else:
        print('Training from the scratch.')
        utils.set_trainable(model, True)
        args.optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)

    args.scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.9)

    if args.evaluate:
        model_nm = args.load_model
        utils.load_net(model_nm, [model])
        model.eval()
        top1, top5, val_loss = eval(model, dev_loader, args)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.4f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.4f}', '5-')))

    if args.train_topkverb:
        print('Model training with topk verb started!')
        args.text_dict = text_dict
        #args.verb_dict = verb_dict
        #args.role_dict = role_dict
        if args.verb_model=='wiseft':
            wiseft = WiseFT()
            verb_model = wiseft.mlp_verb_model

        elif args.verb_model=='verb_mlp':
            constructor = 'build_%s' % args.verb_model
            verb_model = getattr(verb_models, constructor)(args)
            utils.load_net('verb_models/verb_mlp_lr0.001_bs64_nl{}_{}'.format(args.num_verb_layers, args.img_emb_base_verb), [verb_model])
            verb_model.to(device)
            print('Successfully loaded mlp-verb model!')
        # print('Successfully loaded verb model!')
    
        train_with_topkverb(verb_model, model, train_loader, dev_loader,args, topk=3)


    elif args.verbeval:
        #utils.load_net(verb_model_nm, [mlp_verb_model])
        #print('successfully loaded mlp verb model!')
        if len(args.load_model) == 0:
            raise Exception('[pretrained module] not specified')
        role_model_nm = 'trained_models/' + args.load_model
        utils.load_net(role_model_nm, [model])
        print('Successfully loaded Role {} model!'.format(args.model))
        args.text_dict = text_dict
        #args.verb_dict = verb_dict
        #args.role_dict = role_dict
        if args.verb_model=='wiseft':
            wiseft = WiseFT()
            mlp_verb_model = wiseft.mlp_verb_model
            print('Successfully loaded wise-ft verb model!')

        elif args.verb_model=='verb_mlp':
            constructor = 'build_%s' % args.verb_model
            mlp_verb_model = getattr(verb_models, constructor)(args)
            utils.load_net('verb_models/verb_mlp_lr0.001_bs64_nl{}_{}'.format(args.num_verb_layers, args.img_emb_base_verb), [mlp_verb_model])
            mlp_verb_model.to(device)
            print('Successfully loaded mlp-verb model!')


        top1, top5, val_loss = jointeval(mlp_verb_model, model, dev_loader, args, write_to_file = True)

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.4f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.4f}', '5-')))

        top1, top5, val_loss = jointeval(mlp_verb_model, model, test_loader, args, write_to_file = True)

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.4f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.4f}', '5-')))
        


    elif args.test:
        model_nm = args.load_model
        utils.load_net(model_nm, [model])
        model.eval()
        top1, top5, test_loss = eval(model, test_loader, args)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                    utils.format_dict(top1_avg,'{:.4f}', '1-'),
                                                    utils.format_dict(top5_avg, '{:.4f}', '5-')))
    
    elif args.save_noun_predictions:
        model_nm = 'trained_models/' + args.load_model
        utils.load_net(model_nm, [model])
        model.eval()

        with open('data/output/saved_noun_preds/dev.pkl', 'wb') as f:
            noun_preds = save_noun_predictions(model, dev_loader, args)
            pickle.dump(noun_preds, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open('data/output/saved_noun_preds/test.pkl', 'wb') as f:
            noun_preds = save_noun_predictions(model, test_loader, args)
            pickle.dump(noun_preds, f, protocol=pickle.HIGHEST_PROTOCOL)


    elif args.save_verb_and_noun_predictions:
        role_model_nm = 'trained_models/' + args.load_model
        utils.load_net(role_model_nm, [model])
        wiseft = WiseFT()
        verb_model = wiseft.mlp_verb_model
        print('Successfully loaded wise-ft verb model and role model!')
        args.text_dict = text_dict
        #args.verb_dict = verb_dict
        #args.role_dict = role_dict
        model.eval()
        with open('data/output/saved_verbnoun_preds/dev.pkl', 'wb') as f:
            preds = save_both_predictions(model,verb_model, dev_loader, args)
            pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open('data/output/saved_verbnoun_preds/test.pkl', 'wb') as f:
            preds = save_both_predictions(model,verb_model, test_loader, args)
            pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)


    else:
        print('Model training started!')
        train(model, train_loader, dev_loader, args)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()












