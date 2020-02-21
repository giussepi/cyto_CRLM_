import os

class unet_seg_base():
    # configs
    n_classes = 5
    patch_size = 1024
    num_workers =8
    
    ## Model related
    device_ids=[0]
    input_nc = 3
    use_dropout = True
    lr = 0.001
    optim = 'rms'  # training options:  'sdg' || 'rms'
    weight_decay = 1e-4


class unet_seg_v1(unet_seg_base):
    name = 'unet_seg_v1'
    batch_size = 1
    block_type = 'conv'
    
    num_blocks = 2
    num_channels =[64,64,128,128,256]
    #strides = [2,2,2,2,2]
    
    optim = 'rms'
    display_size = 400
    save_epoch = 2
    milestones=[15,30]  # for learning rate adaption
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
    '/DATA_CRLM/Patches/Checkpoints/U_Seg/%s/'%name    

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

class unet_seg_v2(unet_seg_base):
    name = 'unet_seg_v2'
    batch_size = 1
    block_type = 'res'
    
    num_blocks = 2
    num_channels =[64,64,128,128,256]
    #strides = [2,2,2,2,2]
    
    optim = 'sgd'
    display_size = 400
    save_epoch = 2
    milestones=[15,30]  # for learning rate adaption
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
    '/DATA_CRLM/Patches/Checkpoints/U_Seg/%s/'%name    

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
