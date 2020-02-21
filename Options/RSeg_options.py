import os

class res_seg_base():
    # configs
    n_classes = 5
    patch_size = 1024
    num_workers =8

    ## Model related
    device_ids=[0]
    norm = 'instance'
    
    input_nc = 3
    output_nc = 5
    
    use_dropout = True
    n_downsampling = 3
    
    ngf = 64
    init_type = 'normal'
    
    lr = 0.001
    optim = 'rms'  # training options:  'sdg' || 'rms'
    weight_decay = 1e-4
    device_ids=[0]    


class res_seg_v1(res_seg_base):
    name = 'res_seg_v1'
    batch_size = 2   
    n_blocks = 9
    
    optim = 'rms'
    display_size = 400
    save_epoch = 5
    milestones=[30]  # for learning rate adaption
    num_epochs= 50
    
    
    train_path_sample = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Sample/'
    train_path_mask = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Mask/'
    test_path_sample = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Testing/Sample/'
    test_path_mask = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Testing/Mask/'

    save_path = os.path.expanduser('~')+\
    '/DATA_CRLM/Checkpoints/Res_Seg/%s/'%name    

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)


class res_seg_v2(res_seg_base):
    name = 'res_seg_v2'
    batch_size = 3
    n_blocks = 6
    
    optim ='sgd'
    display_size = 100
    save_epoch = 5
    milestones=[30]  # for learning rate adaption
    num_epochs= 50
    
    
    train_path_sample = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Sample/'
    train_path_mask = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Training/Mask/'
    test_path_sample = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Testing/Sample/'
    test_path_mask = os.path.expanduser('~')+\
    '/DATA_CRLM/Patches/Patches_Segment/Synth_Patch_1024_Paired/Testing/Mask/'

    save_path = os.path.expanduser('~')+\
    '/DATA_CRLM/Checkpoints/Res_Seg/%s/'%name    

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
