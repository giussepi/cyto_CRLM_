import os


class udense_v1():
    name='Udense_v1'
    
    down_blocks=(4,4,4,4,4,4)
    up_blocks=(4,4,4,4,4,4)
    bottleneck_layers=4
    growth_rate=16
    out_chans_first_conv=64
    n_classes =11
    
    patch_size = 448
    batch_size = 2
    num_workers = 8   # for HPC only
    
    display_size = 400
    save_epoch = 2
    milestones=[2,5,10]  # for learning rate adaption
    num_epochs= 20
    
    lr = 0.001
    optim = 'rms'  # training options:  'sdg' || 'rms'
    weight_decay = 1e-4
    device_ids=[0]    
    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/All/'
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_448/Test/'
    mask_dir = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Segment/Patches_Syth_mask_448/'
    save_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Checkpoints/Unets/%s/'%name
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    



             