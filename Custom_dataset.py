
from torch.utils.data import Dataset
import os 
import pandas as pd
import torch.nn.functional as TF
from PIL import Image
import numpy as np

class MultiModalData(Dataset):
    def __init__(self, Dir_feat, Dir_images):
        super (MultiModalData, self).__init__()
        self.Dir_feat = Dir_feat
        self.Dir_images = Dir_images
        self.Dir_images_list = os.listdir(self.Dir_images)
        self.width = 324 #1024
        self.height = 324 #683

    def __len__(self):
        data_frame = pd.read_csv(self.Dir_feat)
        len_feat_data = len(data_frame.index)
        # len_images_data = len(self.Dir_images_list)
        return len_feat_data

    def __getitem__(self, index):
        # features
        data_frame = pd.read_csv(self.Dir_feat)
        headers = data_frame.columns.values
        features = headers[3:12]  # columns of the features 
        data_feat = data_frame[features]  # data from those features
        # Normalize 
        data_feat=(data_feat-data_feat.min())/(data_feat.max()-data_feat.min())
           
        data_features  = data_feat.iloc[index]  # features of the sample
        data_features = np.array(data_features, dtype=np.float32)
        label = data_frame[headers[-1]][index]  # label of the sample
        label = np.array(label, dtype= np.float32)
        
        # image
        image = Image.open(os.path.join(self.Dir_images, self.Dir_images_list[index])).convert('RGB')  # image of the sample
        image = image.resize((self.width, self.height))
        image = np.array(image, dtype = np.float32)
        image -= image.min(axis=(0,1,2), keepdims=True)
        image /= image.max(axis=(0,1,2), keepdims=True)        # normalize
        image = np.moveaxis(image, -1,0)
        # print((data_features).shape, image.shape, label.shape)
        return  data_features, image, label



        
