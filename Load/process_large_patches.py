import torch
import numpy as np
import pylab as plt

def process_large_image(model,input_patch,step = 28,out_scale =4,num_classes= 11,patch_size = 448,show=False,cuda_size=None):
    """
    step = 28   #sliding window step size
    out_scale =4   # the output shape of the network
    num_classes= 11   # the number of classes 
    patch_size = 448     # the patch size of the 
    """
    if input_patch.max()>2:
        test_img = input_patch/255.0
    else:
        test_img = input_patch
    tt = torch.from_numpy(((test_img[:,:,(0,1,2)]-np.array([0.485, 0.456, 0.406]))/ np.array([0.229, 0.224, 0.225])).transpose(2,0,1)).float()

    ta =tt.unfold(2,patch_size,step)
    #print(ta.size())
    tb = ta.unfold(1,patch_size,step)
    #print(tb.size())
    tc = tb.permute((1,2,0,3,4))
    #print(tc.shape)
    td = tc.reshape(-1,3,patch_size,patch_size)
    #print(td.shape)
    nx = len(range(0,test_img.shape[0]-patch_size+1,step))
    ny = len(range(0,test_img.shape[1]-patch_size+1,step))
    
    final_tensor = td
    final_result2 = []
    if cuda_size==None:
        for i in range(0,nx):
            test_tensor = final_tensor[i*ny:i*ny+ny]
            out = model(test_tensor.cuda())
            softmax = torch.nn.Softmax2d()
            out2 = softmax(out)
            final_result2.append(out2.detach().cpu().numpy())
            torch.cuda.empty_cache() 
        t = np.array(final_result2).transpose(0,3,1,4,2).reshape(nx*out_scale,ny*out_scale,num_classes)
        torch.cuda.empty_cache()
    else:
        for i in range(0,nx*ny,cuda_size):
            test_tensor = final_tensor[i:i+cuda_size]
            out = model(test_tensor.cuda())
            softmax = torch.nn.Softmax2d()
            out2 = softmax(out)
            if out2.size(0)!=cuda_size:
                tout2 = np.zeros((cuda_size,out2.size(1),out2.size(2),out2.size(3)))
                tout2[:out2.size(0)] = out2.detach().cpu().numpy()
                final_result2.append(tout2)
            else:
                final_result2.append(out2.detach().cpu().numpy())
            torch.cuda.empty_cache()
            
        tt = np.array(final_result2)
        s = tt.shape
        tt = tt.reshape(-1,s[2],s[3],s[4])[:nx*ny].reshape(nx,ny,s[2],s[3],s[4])
        t = tt.transpose(0,3,1,4,2).reshape(nx*out_scale,ny*out_scale,num_classes)
    
    if show:
        plt.subplot(1,2,1)
        plt.imshow(test_img)
        plt.subplot(1,2,2)
        plt.imshow(np.argmax(t[:,:,:],2))
    return t
