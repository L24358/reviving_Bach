import os
import torch
import numpy as np
import torch.nn as nn
import muse.processor as pcr

def KL(mu, logvar):
    return 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def load_model(mdl_name, model, device):
    state = torch.load(os.path.join('./models/', mdl_name+'.pt'))
    model.load_state_dict(state['state_dict'])
    model.float()
    model.to(device)
    train_losses, val_losses = state['train_losses'], state['val_losses'] 
    train_acc, val_acc = state['train_acc'], state['val_acc'] 
    epoch, seeds = state['epoch'], state['seeds']
    return model, train_losses, val_losses, train_acc, val_acc, epoch, seeds

def accuracy(ypred, targ):
    return [int(ypred[i])==int(targ[i]) for i in range(len(ypred))].count(True)/len(ypred)

def train_VAE(model, epochs, lr, train_loader, val_loader, device, 
                seeds, mdl_name, eepoch=10, epoch_start=0,
                train_losses=[], val_losses=[], train_acc=[], val_acc=[],
                save=True, load=False):
    
    # get loss function 
    criterion = nn.MSELoss()
    
    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # load model
    if load:
        model, train_losses, val_losses, train_acc, val_acc, epoch_start, seeds = load_model(mdl_name, model, device)

    for epoch in range(epoch_start, epoch_start+epochs):

        # train model
        model.train()
        batch_losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            recons, mu, logvar = model(batch_data)
            l = criterion(recons, batch_data) + KL(mu, logvar)
            l.backward()
            optimizer.step()
            batch_losses.append(l.item())
        train_losses.append(np.mean(batch_losses))
        train_acc.append(accuracy(recons.view(-1), batch_data.view(-1)))

        # validate model
        model.eval()
        batch_losses = []
        for batch_idx, batch_data in enumerate(val_loader):
            batch_data = batch_data.to(device)
            recons, mu, logvar = model(batch_data)
            l = criterion(recons, batch_data) + KL(mu, logvar)
            batch_losses.append(l.item())
        val_losses.append(np.mean(batch_losses))
        val_acc.append(accuracy(recons.view(-1), batch_data.view(-1)))
        
        # print loss
        if epoch % eepoch == 0:
            print('Epoch: {}\t TLoss: {:.6f}\t VLoss: {:.6f}\t TACC: {:.2f}\t VACC: {:.2f}'\
                  .format(epoch, train_losses[-1], val_losses[-1], train_acc[-1], val_acc[-1]))
    
        # save model
        if save:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
              'train_losses': train_losses, 'val_losses':val_losses, 
              'train_acc':train_acc, 'val_acc':val_acc, 'seeds': seeds}
            torch.save(state, './models/'+mdl_name+'.pt')
        
    return model, train_losses, val_losses, train_acc, val_acc

def train_VAE_2(model, epochs, lr, train_loader, val_loader, device, 
                seeds, mdl_name, eepoch=10, epoch_start=0,
                train_losses=[], val_losses=[], train_acc=[], val_acc=[],
                save=True, load=False, lossfunc='MSE', stdlog=[],
                lda1=1, lda2=1):
    
    # get loss function 
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    
    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # load model
    if load:
        model, train_losses, val_losses, train_acc, val_acc, epoch_start, seeds = load_model(mdl_name, model, device)

    for epoch in range(epoch_start, epoch_start+epochs):
        
        # set std
        if stdlog != []:
            model.std = stdlog[epoch-epoch_start]

        # train model
        model.train()        
        batch_losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            recons, mu, logvar, prob = model(batch_data)
            if lossfunc == 'MSE':
                l = criterion1(recons, batch_data) + KL(mu, logvar)
            elif lossfunc == 'CE':
                l = criterion2(prob, batch_data.squeeze().long()) + KL(mu, logvar)
            elif lossfunc == 'both':
                l = criterion1(recons, batch_data) + lda1*criterion2(prob, batch_data.squeeze().long()) + lda2*KL(mu, logvar)
            l.backward()
            optimizer.step()
            batch_losses.append(l.item())
        train_losses.append(np.mean(batch_losses))
        train_acc.append(accuracy(recons.view(-1), batch_data.view(-1)))

        # validate model
        model.eval()
        batch_losses = []
        for batch_idx, batch_data in enumerate(val_loader):
            batch_data = batch_data.to(device)
            recons, mu, logvar, prob = model(batch_data)
            if lossfunc == 'MSE':
                l = criterion1(recons, batch_data) + lda2*KL(mu, logvar)
            elif lossfunc == 'CE':
                l = criterion2(prob, batch_data.squeeze().long()) + lda2*KL(mu, logvar)
            elif lossfunc == 'both':
                l = criterion1(recons, batch_data) + lda1*criterion2(prob, batch_data.squeeze().long()) + lda2*KL(mu, logvar)
            batch_losses.append(l.item())
        val_losses.append(np.mean(batch_losses))
        val_acc.append(accuracy(recons.view(-1), batch_data.view(-1)))
        
        # print loss
        if epoch % eepoch == 0:
            print('Epoch: {}\t TLoss: {:.6f}\t VLoss: {:.6f}\t TACC: {:.2f}\t VACC: {:.2f}'\
                  .format(epoch, train_losses[-1], val_losses[-1], train_acc[-1], val_acc[-1]))
   
        # save model
        if save:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
              'train_losses': train_losses, 'val_losses':val_losses, 
              'train_acc':train_acc, 'val_acc':val_acc, 'seeds': seeds}
            torch.save(state, './models/'+mdl_name+'.pt')
        
    return model, train_losses, val_losses, train_acc, val_acc

def keep_lowest(mdl_name, model, device, deter='train', metric='acc', delete=True):
    dic = {}
    for file in os.listdir('./models/'):
        if mdl_name in file:
            name = file.split('.pt')[0]
            model, train_losses, val_losses, train_acc, val_acc, epoch, seeds =\
                load_model(name, model, device)
                
            if (deter=='train') and (metric=='loss'): judge = train_losses[-1]
            elif (deter=='val') and (metric=='loss'): judge = val_losses[-1]
            elif (deter=='train') and (metric=='acc'): judge = -train_acc[-1]
            elif (deter=='val') and (metric=='acc'): judge = -val_acc[-1]
            dic[name] = judge
            
    mdl_min = [k for k, v in sorted(dic.items(), key=lambda item: item[1])][0]
    model, train_losses, val_losses, train_acc, val_acc, epoch, seeds = load_model(mdl_min, model, device)
    state = {'epoch': epoch, 'state_dict': model.state_dict(),
              'train_losses': train_losses, 'val_losses':val_losses, 
              'train_acc':train_acc, 'val_acc':val_acc, 'seeds': seeds}
    torch.save(state, './models/'+mdl_name+'.pt')
    
    if mdl_name in dic.keys(): dic.pop(mdl_name)
    if delete:
        for key in dic.keys(): os.remove(os.path.join('models', key+'.pt'))
    return model, train_losses, val_losses, train_acc, val_acc, epoch, seeds
            