from utils.loss_functions import CustomSimLoss, ContrastiveLoss, CustomMinSimLoss

def get_embed_loss(pred_noun_embedding,label_embeddings,target_mask, mask, role_embeddings, args, label_strs=None):
    
    loss_function = args.embed_loss_fn

    if len(label_embeddings.shape)==3 and loss_function == 'SIM_COS':
        loss_fn = CustomMinSimLoss('cos')
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,target_mask)
        return embed_loss
    
    elif len(label_embeddings.shape)==3 and loss_function == 'SIM_EXP_COS':
        loss_fn = CustomMinSimLoss('exp_cos')
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,target_mask)
        return embed_loss
    
    if loss_function == None:
        embed_loss = 0
    elif loss_function == 'SIM_COS':
        loss_fn = CustomSimLoss('cos')
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,target_mask)
    elif loss_function == 'SIM_EXP_COS':
        loss_fn = CustomSimLoss('exp_cos')
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,target_mask)
    elif loss_function == 'SIM_JAC':
        loss_fn = CustomSimLoss('jac')
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,target_mask)

    elif loss_function == 'SIM_ROLE_NOUN':
        loss_fn = CustomSimLoss('ccrn')
        embed_loss = loss_fn(role_embeddings, pred_noun_embedding,mask) # TO FIX
    elif loss_function == 'SIM_COS+SIM_ROLE_NOUN':
        loss_fn_1 = CustomSimLoss('cos')
        loss_fn_2 = CustomSimLoss('ccrn')
        embed_loss = loss_fn_1(pred_noun_embedding,label_embeddings,target_mask) + loss_fn_2(role_embeddings, pred_noun_embedding,mask)
    elif loss_function == 'SIM_JAC+SIM_ROLE_NOUN':
        loss_fn_1 = CustomSimLoss('jac')
        loss_fn_2 = CustomSimLoss('ccrn')
        embed_loss = loss_fn_1(pred_noun_embedding,label_embeddings,target_mask) + loss_fn_2(role_embeddings, pred_noun_embedding,mask)

    elif loss_function == 'DIFF_COV':
        loss_fn = CustomSimLoss('cov')
        embed_loss = loss_fn(label_embeddings, pred_noun_embedding,target_mask)
    elif loss_function == 'DIFF_COV+SIM_COS':
        loss_fn_1 = CustomSimLoss('cov')
        loss_fn_2 = CustomSimLoss('cos')
        embed_loss = loss_fn_1(label_embeddings, pred_noun_embedding,target_mask) + loss_fn_2(pred_noun_embedding, label_embeddings,target_mask)
    elif loss_function == 'DIFF_COV+SIM_JAC':
        loss_fn_1 = CustomSimLoss('cov')
        loss_fn_2 = CustomSimLoss('jac')
        embed_loss = loss_fn_1(label_embeddings, pred_noun_embedding,target_mask) + loss_fn_2(pred_noun_embedding, label_embeddings,target_mask)
    elif loss_function == 'DIFF_COV+SIM_ROLE_NOUN':
        loss_fn_1 = CustomSimLoss('cov')
        loss_fn_2 = CustomSimLoss('ccrn')
        embed_loss = loss_fn_1(label_embeddings, pred_noun_embedding,target_mask) + loss_fn_2(role_embeddings, pred_noun_embedding,mask) # TO FIX

    elif loss_function == 'CNT_EXP_COS':
        loss_fn = ContrastiveLoss('exp_cos', num_negatives=args.num_negatives)
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,label_strs, target_mask)
    elif loss_function == 'CNT_COS':
        loss_fn = ContrastiveLoss('cos', num_negatives=args.num_negatives)
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,label_strs, target_mask)
    elif loss_function == 'CNT_JAC':
        loss_fn = ContrastiveLoss('jac')
        embed_loss = loss_fn(pred_noun_embedding,label_embeddings,label_strs, target_mask)

    elif loss_function == 'DIFF_COV+CNT_COS':
        loss_fn_1 = CustomSimLoss('cov')
        loss_fn_2 = ContrastiveLoss('cos')
        embed_loss = loss_fn_1(label_embeddings, pred_noun_embedding, target_mask) + loss_fn_2(pred_noun_embedding, label_embeddings,label_strs, target_mask)
    elif loss_function == 'DIFF_COV+CNT_EXP_COS':
        loss_fn_1 = CustomSimLoss('cov')
        loss_fn_2 = ContrastiveLoss('exp_cos')
        embed_loss = loss_fn_1(label_embeddings, pred_noun_embedding, target_mask) + loss_fn_2(pred_noun_embedding, label_embeddings,label_strs, target_mask)

    elif loss_function == 'DIFF_COV+CNT_JAC':
        loss_fn_1 = CustomSimLoss('cov')
        loss_fn_2 = ContrastiveLoss('jac')
        embed_loss = loss_fn_1(label_embeddings, pred_noun_embedding, target_mask) + loss_fn_2(pred_noun_embedding, label_embeddings,label_strs, target_mask)
    else:
        raise ValueError('Invalid loss function')
    
    return embed_loss
