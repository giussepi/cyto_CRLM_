import torch
import time
import copy
import logging

def train_model(model, dataloaders,criterion, optimizer, scheduler, num_epochs=25,\
                display_size=100,save_epoch=5,save_path='./'):
    since = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    #model.cuda()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_index,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs[:,:3,:,:])
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if len(labels.data.size())==2:
                    running_corrects += torch.sum(preds == labels.data.view(-1,1,1))*1.0/(outputs.shape[2]*outputs.shape[3])
                else:
                    running_corrects += torch.sum(preds == labels.data.long())*1.0/(outputs.shape[2]*outputs.shape[3])

                if phase == 'train' and  batch_index% display_size==0:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss/(batch_index+1)/inputs.shape[0], running_corrects.double()/(batch_index+1)/inputs.shape[0]))
                    logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss/(batch_index+1)/inputs.shape[0], running_corrects.double()/(batch_index+1)/inputs.shape[0]))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            

            if (epoch+1)%save_epoch == 0:
                torch.save(model.state_dict(),save_path + 'PatchCNN%03d.pth'%(epoch+1))
                print("model saved %d"%(epoch+1))

            
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.cpu().state_dict(),save_path+'PatchCNN_best.pth')
    return model