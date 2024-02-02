import torch
import numpy as np
from thop import profile

def measure_inference_time(model, args, device='cuda', repetitions=300):
    model.to(device)
    model.eval()
    if args.model == 'xtf':
        if args.img_emb_base == 'vit-b16':
            num_patches = 196 
        elif args.img_emb_base == 'vit-l14':
            num_patches = 256
        elif args.img_emb_base == 'vit-l14-336':
            num_patches = 576
        elif args.img_emb_base == 'vit-b32':
            num_patches = 49
        img_embeddings_shape = [args.batch_size,num_patches, args.image_dim]
    else:
        img_embeddings_shape = [args.batch_size, args.image_dim]
    verb_embeddings_shape = [args.batch_size, args.text_dim]
    role_embeddings_shape = [args.batch_size * args.encoder.max_role_count, args.text_dim]
    #if args.model=='xtf':
    mask_shape = [args.batch_size, args.encoder.max_role_count]
    #else:
        # is this correct for transformer???
        #mask_shape = [args.batch_size * args.encoder.max_role_count, args.proj_dim]
    # Generate dummy data
    img_embeddings = torch.randn(*img_embeddings_shape, dtype=torch.float).to(device)
    verb_embeddings = torch.randn(*verb_embeddings_shape, dtype=torch.float).to(device)
    role_embeddings = torch.randn(*role_embeddings_shape, dtype=torch.float).to(device)
    mask = torch.randn(*mask_shape, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = model(img_embeddings, verb_embeddings, role_embeddings, mask)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(img_embeddings, verb_embeddings, role_embeddings, mask)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)

    return mean_time, std_time


def measure_flops(model,args, device='cuda'):
    model.to(device)
    model.eval()

    if args.model == 'xtf':
        if args.img_emb_base == 'vit-b16':
            num_patches = 196 
        elif args.img_emb_base == 'vit-l14':
            num_patches = 256
        elif args.img_emb_base == 'vit-l14-336':
            num_patches = 576
        elif args.img_emb_base == 'vit-b32':
            num_patches = 49
        img_embeddings_shape = [args.batch_size,num_patches, args.image_dim]
    else:
        img_embeddings_shape = [args.batch_size, args.image_dim]
    verb_embeddings_shape = [args.batch_size, args.text_dim]
    role_embeddings_shape = [args.batch_size * args.encoder.max_role_count, args.text_dim]
    mask_shape = [args.batch_size,  args.encoder.max_role_count]

    # Generate dummy data
    img_embeddings = torch.randn(*img_embeddings_shape, dtype=torch.float).to(device)
    verb_embeddings = torch.randn(*verb_embeddings_shape, dtype=torch.float).to(device)
    role_embeddings = torch.randn(*role_embeddings_shape, dtype=torch.float).to(device)
    mask = torch.randn(*mask_shape, dtype=torch.float).to(device)

    macs, params = profile(model, inputs=(img_embeddings, verb_embeddings, role_embeddings, mask))
    flops = macs * 2  # converting MACs to FLOPs
    gflops = flops / (10**9)
    return gflops, params



def measure_inference_time_verb(model, args, device='cuda', repetitions=300):
    model.to(device)
    model.eval()

    img_embeddings_shape = [args.batch_size, args.image_dim]
    img_embeddings = torch.randn(*img_embeddings_shape, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    for _ in range(10):
        _ = model(img_embeddings)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(img_embeddings)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)

    return mean_time, std_time


def measure_flops_verb(model,args, device='cuda'):
    model.to(device)
    model.eval()


    img_embeddings_shape = [args.batch_size, args.image_dim]

    # Generate dummy data
    img_embeddings = torch.randn(*img_embeddings_shape, dtype=torch.float).to(device)

    macs, params = profile(model, inputs=(img_embeddings,))
    flops = macs * 2  # converting MACs to FLOPs
    gflops = flops / (10**9)
    return gflops, params
