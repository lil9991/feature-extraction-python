import torchextractor as tx
import torch.nn as nn
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.io as sio
import argparse
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#%%
test_transforms = transforms.Compose([
    
   transforms.Resize((224,224)),
  # transforms.Grayscale(num_output_channels=3),
   #transforms.Resize((299,299)), #xception
   transforms.ToTensor(),
   transforms.Normalize([0.5629, 0.5454, 0.6433], [0.1347, 0.1304, 0.1230])
  
  #fold1:   [0.4983, 0.4919, 0.5838], [0.1132, 0.1159, 0.0999]
  #fold2: tensor([0.5629, 0.5454, 0.6433]) , tensor([0.1347, 0.1304, 0.1230])
])
#%%

train_dataset = datasets.ImageFolder('Train', transform= test_transforms)
print(len(train_dataset))
test_dataset  = datasets.ImageFolder('Test' , transform= test_transforms)
print(len(test_dataset))

#%%
batch_size=32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size, 
                                          shuffle= False)  

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size,
                                          shuffle= False)

#%%
from resnet_pytorch import ResNet 
model = ResNet.from_pretrained('resnet18', num_classes=5)
model.load_state_dict( torch.load("model/resnet18-model_latest.pth") )

#%%

#model = models.resnext50_32x4d(pretrained=True) 
#model.fc = torch.nn.Linear(2048, 5)
#model.load_state_dict( torch.load("model/resnext50-model_latest.pth") )
#%%

#model = models.mobilenet_v2(pretrained=True)
#model.fc = torch.nn.Linear(1280, 5)
#model.load_state_dict( torch.load("Model/mobilenet_v2-model_latest.pth") )


#%%
import torchvision.models as models
#model = models.resnet50(pretrained=True)
#model.fc = torch.nn.Linear(2048, 5)
#model.load_state_dict( torch.load("model/resnet50-model_latest.pth") )

#%%
from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b4') 
#model.fc = torch.nn.Linear(1000, 5)
#model.load_state_dict( torch.load("model/efficientnet-b4-model_latest.pth") )


#%%

model.eval()
print(model)
#%%
for name, layer in model.named_modules():
    
    if isinstance(layer, torch.nn.Linear):
     print(name)
#%%
modelx = tx.Extractor(model, ["_avg_pooling"])

#dummy_input = torch.rand(64 ,3, 299, 299)
#model_output, features = modelx(dummy_input)

#feature_shapes = {name: f.shape for name, f in features.items()}
#print(feature_shapes)

#A=features['Mixed_6e.branch_pool'].detach().numpy()
#result = A[:, :, 0,0]
#print(result.shape)

#%%%
cnt = 0

for X, y in train_loader: 
    X = X.cuda()
    y = y.cuda()
    
    modelx.eval().cuda()
    
    with torch.no_grad():
        model_output, features = modelx(X)
        my_features = features['_avg_pooling']
        A=my_features.cpu().detach().numpy()
        result = A[:, :, 0,0]
        #print(result.shape)
        temp= np.c_[result ,y.cpu()]
        if cnt==0:
            X_trn=temp
            cnt=cnt+1
        else:
            X_trn=np.r_[X_trn,temp]
            cnt=cnt+1
    
#%%
cnt = 0

for X, y in test_loader: 
    X = X.cuda()
    y = y.cuda()
    
    modelx.eval().cuda()
    
    with torch.no_grad():
        model_output, features = modelx(X)
        my_features = features['_avg_pooling']
        A=my_features.cpu().detach().numpy()
        result = A[:, :, 0,0]
        #print(result.shape)
        temp= np.c_[result ,y.cpu()]
        if cnt==0:
            X_tst=temp
            cnt=cnt+1
        else:
            X_tst=np.r_[X_tst,temp]
            cnt=cnt+1
   
#%%   


x_test = {"X_tst": X_tst, "label": "X_tst"}
x_train = {"X_trn": X_trn, "label": "X_trn"}

#savemat("resnet18-model-features.mat", {"X_trn": x_train,"X_tst ": x_test })
#%%

#load mat_file
mat_file = sio.loadmat("resnet18-model-features.mat")
print(mat_file)