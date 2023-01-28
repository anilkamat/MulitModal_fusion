from Custom_dataset import MultiModalData
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch 


random_seed = 1024
Dir_feat = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\ManualFeatures_sorted.csv'
Dir_images = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\Images_eye'

#feat, images, label = 
samples = MultiModalData(Dir_feat, Dir_images)

def get_loader(Dir_feat, Dir_images, batch_size, shuffle_dataset = True, split_train = 0.8):
    # indices of train and test dataset
    samples = MultiModalData(Dir_feat, Dir_images) # ((data_features, data_images),labels)
    size_data = len(samples)
    split = int(split_train*size_data)
    indices = list(range(size_data))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[0:split], indices[split:]
    # create PT data sampler and loader 
    train_sampler, test_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(test_indices)

    train_loader = DataLoader(samples, batch_size= batch_size, sampler= train_sampler)
    test_loader = DataLoader(samples, batch_size= batch_size, sampler= test_sampler)

    return train_loader, test_loader
    #train_loader = Dataloader(feat, images, batch_size)

# train_loader, test_loader = get_loader(Dir_feat, Dir_images, batch_size= 16, shuffle_dataset = True, split_train = 0.8)

# num_epochs = 1
# for epoch in range(num_epochs):
#     # Train:   
#     for batch_index, (features,images, labels) in enumerate(train_loader):
#         print(batch_index, features.shape, images.shape, labels)
