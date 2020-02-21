import torch
import time
import copy
import logging

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25,\
                display_size=100,save_epoch=5,save_path='./'):
    since = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}


    #num_epochs=10
    #scheduler = exp_lr_scheduler
    #optimizer = optimizer_ft
    #display_size = 100

    #best_model_wts = copy.deepcopy(model.state_dict())
    train_loss = []
    train_acc = []
    eval_loss = []
    eval_acc = []
    
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
                inputs = inputs.cuda().float()
                labels = labels.cuda().long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).float()/(outputs.shape[2]*outputs.shape[3])
                
                train_loss.append(running_loss/(batch_index+1)/inputs.shape[0])
                train_acc.append(running_corrects.double()/(batch_index+1)/inputs.shape[0])
                
                if phase == 'train' and  batch_index% display_size==0:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, running_loss/(batch_index+1)/inputs.shape[0], running_corrects.double()/(batch_index+1)/inputs.shape[0]))
                
            
            torch.save(train_loss,save_path + 'train_loss.pth')
            torch.save(train_acc,save_path + 'train_acc.pth')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if phase=='eval':
                eval_loss.append(epoch_loss)
                eval_acc.append(eval_acc)
                
            torch.save(epoch_loss,save_path + 'eval_loss.pth')
            torch.save(epoch_acc,save_path + 'eval_acc.pth')
            
            if (epoch+1)%save_epoch == 0:
                torch.save(model.state_dict(),save_path + 'unet_%03d.pth'%(epoch+1))
                print("model saved %d"%(epoch+1))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))


