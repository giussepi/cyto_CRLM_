import os

device_ids=[0]

class base_option():
    batch_size = 32
    patch_size = 448
    num_workers = 32

    device_ids = [0]
    
    lr = 0.01
    optim = 'rms'
    weight_decay = 1e-3
    dropout_rate = 0.3
    milestones = [20,50,80]
    save_epoch = 30
    display_size = 200
    
    num_epochs = 100
    momentum = 0.9
    gamma=0.1
    
    mean = [0.485, 0.456, 0.406]
    std = [0.5, 0.5, 0.5]
    
    color_aug_param_train = [0.2,0.2,0.1,0.2]
    color_aug_param_test = [0.1,0.1,0.1,0.1]


        
class pcnn_res34_448(base_option):
    model_name = 'pcnn_res34_448'
    
    num_layers = [3,4,6,3]  # res34
    patch_size = 448
                             
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~')+'/DATA_CRLM/Checkpoints/PatchCNN/%s/'%model_name
