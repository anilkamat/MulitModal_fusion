import numpy as np
from Model import fusion_model
from Utils import get_loader
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def compute_loss(y_hat, y):
    return nn.BCELoss()(y_hat, y)

Dir_feat = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\ManualFeatures_sorted.csv'
Dir_images = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\Images_eye'
batch_size= 64
train_loader, test_loader = get_loader(Dir_feat, Dir_images, batch_size, shuffle_dataset = True, split_train = 0.8)

losses = [] # stores loss of each epochs
val_losses = [] # stores validation loss
xaxis_val = []
num_epochs = 10

model_full = fusion_model()
print(model_full)
optimizer = optim.Adam(model_full.parameters(), lr = 0.0001, weight_decay=1e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)
for epoch in range(num_epochs):
    loop = tqdm(train_loader)
    loss = []  # stores loss of each batch
    # Train:   
    for batch_index, (features, images, labels) in enumerate(loop):
        input = (features, images)
        output = model_full(input)
        output = torch.squeeze(output)
        error = compute_loss(output, labels)
        loss.append(error.detach().numpy()/batch_size)    

        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    scheduler.step()
    losses.append(sum(loss)/(len(loss)))
    print(f'{epoch} epcoh loss :{sum(loss)/(len(loss))}')

    # validation: 
    val_loss = []  # stores loss of each batch
    if epoch%5 ==0: 
        for batch_index, (features, images, labels) in enumerate(test_loader):
            input = (features, images)
            model_full.eval() # put the model to the evaluation mode
            with torch.no_grad():
                output = model_full(input)
            output = torch.squeeze(output)
            error = compute_loss(output, labels)
            val_loss.append(error.detach().numpy()/batch_size)    
        val_losses.append(sum(val_loss)/(len(val_loss)))
        xaxis_val.append(epoch)
        print(f'{epoch} Validation loss :{sum(val_losses)/(len(val_loss))}')
    model_full.train() # Turn back the model to evaluation mode

plt.figure()
plt.plot(losses,linewidth = 1.5)
plt.plot(xaxis_val,val_losses, linewidth = 1.5)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train loss','validation loss'])
plt.title('Convergence history')

# testing
print('testing ... ')
loop = tqdm(test_loader)
test_loss = []
output_all = []
labels_all = []
model_full.eval()
for batch_index, (features, images, labels) in enumerate(loop):
        input = (features, images)
        #optimizer.zero_grad()
        with torch.no_grad():
            output = model_full(input)
            output = torch.squeeze(output)
            
            print(batch_index, output.shape, labels.shape)
            error = compute_loss(output, labels)
            test_loss.append(error)
            output_all.append(output)
            labels_all.append(labels)
print('Test loss: ',sum(test_loss))
print(batch_index, math.ceil(output_all), labels_all)

cm = confusion_matrix(labels_all, output_all)
# ConfusionMatrixDisplay(cm).plot()



