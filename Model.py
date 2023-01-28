import torch     
import torchvision
import torch.nn as nn
import torch.nn.functional as TF
import torch.optim as optim
import tqdm as tqdm
from Utils import get_loader
import matplotlib.pyplot as plt
#from torchsummary import summary 


class model_features(nn.Module):
    def __init__ (self,in_channels_size, out_channels_size):
        super (model_features, self).__init__()
        self.inchannels_size = in_channels_size
        self.outchannels_size = out_channels_size
        out_L1_size = 6
        out_L2_size = 3

        self.L1 = nn.Linear(self.inchannels_size, out_L1_size)
        self.L2 = nn.Linear(out_L1_size, out_L2_size)

    def forward(self, input_features):
        out_L1 = self.L1(input_features)
        out_L1 = TF.elu(out_L1)
        out_L2 = self.L2(out_L1)
        out_L2 = TF.elu(out_L2)
        
        return out_L2

class model_image(nn.Module):
    def __init__(self, inchannel_size, outchannels_size):
        super (model_image, self).__init__()
        self.inchannels = inchannel_size
        self.outchannels = outchannels_size
        out_L1_size = 64
        self.L3_input_size = 4096
        out_L2_size = 64
        out_L22_size = 64
        out_L3_size = 30
        
        self.L1 = nn.Conv2d(self.inchannels, out_L1_size, kernel_size= 16, stride= 8)
        self.L2 = nn.Conv2d(out_L1_size, out_L2_size, kernel_size= 4, stride= 2)
        self.L22 = nn.Conv2d(out_L2_size, out_L22_size, kernel_size= 4, stride= 2)
        self.L3 = nn.Linear(self.L3_input_size, out_L3_size)

    def forward(self, input_mat):
        out_L1 = self.L1(input_mat)
        out_L1 = TF.elu(out_L1)
        out_L2 = self.L2(out_L1)
        
        out_L2 = TF.elu(out_L2)
        out_L22 = self.L2(out_L2)

        m = nn.Flatten()
        out_L22 = m(out_L22)
        #print('flatten: ',out_L22.shape)
        out_L22 = TF.elu(out_L22)
        out_L3 = self.L3(out_L22)
        out_L3 = TF.elu(out_L3)

        return out_L3

class decoder(nn.Module):
    def __init__(self, inchannel_size, outchannel_size):
        super (decoder, self).__init__()
        self.inchannel = inchannel_size
        self.outchannel = outchannel_size
        out_L1_size = 16
        out_L2_size = 8
        out_L3_size = 1

        self.L1 = nn.Linear(self.inchannel, out_L1_size)
        self.L2 = nn.Linear(out_L1_size, out_L2_size)
        self.L3 = nn.Linear(out_L2_size, out_L3_size)
    def forward(self, input):
        out_L1 = self.L1(input)
        out_L1 = TF.elu(out_L1)
        out_L2 = self.L2(out_L1)
        out_L2 = TF.elu(out_L2)
        out_L3 = self.L3(out_L2)
        out_L3 = torch.sigmoid(out_L3)
        
        #print('model output: ',out_L3)

        return out_L3

class fusion_model(nn.Module):
    def __init__(self):
        super (fusion_model, self).__init__()
        
        self.model_feat = model_features(9, 3)
        self.model_img = model_image(3,1)
        self.decoder = decoder(inchannel_size= 33, outchannel_size= 1)

    def forward(self,input):
        input_features = input[0]
        input_image = input[1]
        out_features_model = self.model_feat(input_features) 
        out_image_model = self.model_img(input_image)
        fused_features = torch.concat((out_features_model, out_image_model), dim = 1)  # fusion
        output = self.decoder(fused_features)
        # print('output from model_fusion: ', output, output.shape)
        return output

# test the model

# input = (torch.randn(16, 9),torch.randn(16, 3, 324, 324))
# model_full = fusion_model()
# #summary(model_full, ((1, 9), (1, 3, 683, 1024)))
# print(model_full)
# output = model_full(input)
