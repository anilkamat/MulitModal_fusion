import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as TF
import matplotlib.pyplot as plt

# from Custom_dataset import MultiModalData
from Utils import get_loader

batch_size = 16

# import data
data = pd.read_csv(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\ManualFeatures_sorted.csv')
Dir_feat = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\ManualFeatures_sorted.csv'
header = data.columns.values
print(header[3:12])
features = data[header[3:12]]
labels = data[header[-1]]
#filename_img1 = os.listdir(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\diaret1')
Dir_images = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\Images_eye'
Dir_img1 = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\diaret1'
#filename_img2 = os.listdir(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\normal1')
Dir_img2 = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\normal1'
# img1 = Image.open(os.path.join(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\MultiModal_fusion\Data\data\diaret1',filename_img1[0]))
# img1_array = np.array(img1)
#img1.show()


# feat, image1, image2 = MultiModalData(Dir_feat, Dir_images)

# get_loader(Dir_feat, Dir_images,batch_size)

