import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm_notebook as tqdm
from skimage.transform import resize
from skimage.feature import canny

# Convolutional neural network (two convolutional layers)
class RestoreNet(nn.Module):
    def __init__(self):
        super(RestoreNet, self).__init__()
        
        '''nn.ConvTranspose2d(16,8,kernel_size=10,stride=2,padding=4),
            nn.BatchNorm2d(8),
            nn.ReLU(),'''
        
        ndf = 16
        nlayer = 3
        
        self.nlayer = nlayer
        self.down = nn.AvgPool2d(2, stride=2, padding=0)
        
        m0 = nn.Sequential(
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(1, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.ReLU(),
            
                    nn.ReflectionPad2d(2),
                    nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=0), #100
                    nn.BatchNorm2d(ndf),
                    nn.Sigmoid(),
                    )
        self.m0 = m0
                 
        
    def forward(self, x):
        return self.m0(x)
            
        
    
    def predict(self,x):
        return self.m0(x)
    

# Train the model
def train(model,train_loader,eval_loader,criterion,optimizer,num_epochs,device,state):
    model.train()
    total_step = len(train_loader)
    
    state["best_score"] = 100
    stop_ctr = 0
    
    tloss = 0
    vloss = 0
    vloss_t = 0
    
    losses = []
    
    for epoch in range(num_epochs):
        
        tloss = []
        
        for i, (images,labels) in enumerate(train_loader):
            
            images = images.to(device)
            
            '''labels = labels.permute(0, 2, 3, 1).view(-1)'''
            labels = labels.to(device)
            # Forward pass
            o = model(images)
            
            loss = criterion(o, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tloss.append(loss.item())
        
        tloss = np.mean(tloss)
        vloss = val(model,eval_loader,criterion,device)
        
        losses.append((tloss,vloss))
        
        print ('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.5f}/{:.5f} ' 
                   .format(epoch+1, num_epochs, i+1, total_step, tloss, vloss))
        
        if epoch>state['min_epoch']:
            if vloss<state["best_score"]:
                stop_ctr = 0
                state['best_score'] = vloss
                checkpoint = {
                        'state_dict': model.state_dict(),
                        'state': state
                }
                torch.save(checkpoint, state['save_path'])
            elif vloss>vloss_t:
                stop_ctr+=1
                if stop_ctr>state['max_stop']:
                    break
        
        vloss_t = vloss
        
        
        
    return losses
        

# Test the model
def val(model,eval_loader,criterion,device):
    with torch.no_grad():
        losses = []
        for (images,labels) in eval_loader:
            images = images.to(device)
            
            '''labels = labels.permute(0, 2, 3, 1).view(-1)'''
            labels = labels.to(device)
            # Forward pass
            
            o = model(images)
            
            loss = criterion(o, labels)
                        
            losses.append(loss.item())
            
        return np.mean(losses)


def predict(model,im,device):
    assert len(im.shape) == 2 
       
    h,w = im.shape
    
    mhw = max(h,w)
    
    if (mhw<=32):
        width = 32
    elif (mhw>=512):
        width = 512
    else:
        width = int(np.ceil(mhw/8))*8
    
    
    
    xs = []
    for r in range(0,h,width):
        for c in range(0,w,width):
            
            h1 = min([width,h-r])
            w1 = min([width,w-c])
            
            r1,r2 = r,r+width
            c1,c2 = c,c+width
            
            if h1<width:
                r1,r2 = max([0,h-width]),h
               
            if w1<width:
                c1,c2 = max([0,w-width]),w
            
                
            x = np.zeros((width,width),dtype=np.float32)

            x[0:r2-r1,0:c2-c1] = im[r1:r2,c1:c2]
            
            xs.append(x)
    
    ys = []
    
    for x in xs:
        x = np.expand_dims(np.expand_dims(x,0),0)
        x = torch.tensor(x,dtype=torch.float32).to(device)
        
        y = model.predict(x).detach().cpu().numpy()
        ys.append(y[0,0,:,:])
        
    imo = np.zeros_like(im)
    id_ = 0
    
    for r in range(0,h,width):
        for c in range(0,w,width):
            h1 = min([width,h-r])
            w1 = min([width,w-c])
            
            r1,r2 = r,r+width
            c1,c2 = c,c+width
            
            if h1<width:
                r1,r2 = max([0,h-width]),h
               
            if w1<width:
                c1,c2 = max([0,w-width]),w
            
                
            x = np.zeros((width,width),dtype=np.float32)

            imo[r1:r2,c1:c2] = ys[id_][0:r2-r1,0:c2-c1]
                
            id_+=1
            
    return imo

