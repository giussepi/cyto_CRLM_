import os

class base_option():
    batch_size = 32
    num_workers = 16
    
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
    
    

class alex_v1(base_option):
    model_name = 'alexnet_v1_224'
    base_name = 'alexnet'
    patch_size = 224
    
    device_ids = [0]
    
    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name

    
class alex_v2(base_option):
    model_name = 'alexnet_v2_448'
    base_name = 'alexnet'
    patch_size = 448

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name
    


class vgg16_v1(base_option):
    model_name = 'vgg16_v1_224'
    base_name = 'vgg16'
    patch_size = 224
    
    device_ids = [0]
    
    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name

        
class vgg16_v2(base_option):
    model_name = 'vgg16_v2_448'
    base_name = 'vgg16'
    patch_size = 448
    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name
    

    

class res18_v1(base_option):
    model_name = 'res18_v1_224'
    base_name = 'res18'
    patch_size = 224

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name
    

        
class res18_v2(base_option):
    model_name = 'res18_v2_448'
    base_name = 'res18'
    patch_size = 448

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name

    
class res34_v1(base_option):
    model_name = 'res34_v1_224'
    base_name = 'res34'
    patch_size = 224

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name

        
class res34_v2(base_option):
    model_name = 'res34_v2_448'
    base_name = 'res34'
    patch_size = 448

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name
    

    
class res50_v1(base_option):
    model_name = 'res50_v1_224'
    base_name = 'res50'
    patch_size = 224

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name

        
class res50_v2(base_option):
    model_name = 'res50_v2_448'
    base_name = 'res50'
    patch_size = 448

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name
    

    

    
class dense_v1(base_option):
    model_name = 'dense_v2_224'
    base_name = 'dense'
    patch_size = 224

    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name
    

    

        
class dense_v2(base_option):
    model_name = 'dense_v2_448'
    base_name = 'dense'
    patch_size = 448
    
    device_ids = [0]

    
    train_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/All/'%patch_size
    test_path = os.path.expanduser('~')+'/DATA_CRLM/Patches/Patches_Level0/Patches_%s/Test/'%patch_size
    save_path = os.path.expanduser('~') +'/DATA_CRLM/Patches/Checkpoints/TCP/%s/'%model_name

    
