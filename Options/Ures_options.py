import os


class ures_v1():
    name='Ures_v1'
    
    n_blocks = 2
    block_type = 'res'
    num_features = [64,96,128,192,256,384]
    dropout = 0.5
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
    
  