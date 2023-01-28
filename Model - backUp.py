import torch     
import torchvision
import torch.nn as nn
import torch.nn.functional as TF


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
        out_L1 = TF.relu(out_L1)
        out_L2 = self.L2(out_L1)
        out_L2 = TF.relu(out_L2)
        
        return out_L2

class model_image(nn.Module):
    def __init__(self, inchannel_size, outchannels_size):
        super (model_image, self).__init__()
        self.inchannels = inchannel_size
        self.outchannels = outchannels_size
        out_L1_size = 1
        self.L2_input_size = 672
        out_L2_size = 30
        
        self.L1 = nn.Conv2d(self.inchannels, out_L1_size, kernel_size= 32, stride= 32)
        self.L2 = nn.Linear(self.L2_input_size, out_L2_size)

    def forward(self, input_mat):
        out_L1 = self.L1(input_mat)
        out_L1 = torch.squeeze(out_L1)
        m = nn.Flatten()
        out_L1 = m(out_L1)
        self.L2_input_size = out_L1.shape[1]
        out_L1 = TF.relu(out_L1)
        out_L2 = self.L2(out_L1)
        out_L2 = TF.relu(out_L2)

        return out_L2

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
        out_L1 = TF.relu(out_L1)
        out_L2 = self.L2(out_L1)
        out_L2 = TF.relu(out_L2)
        out_L3 = self.L3(out_L2)
        out_L3 = TF.elu(out_L3)

        return out_L3

class fusion_model(nn.Module):
    def __init__(self, model_features, model_image, decoder):
        super (fusion_model, self).__init__()
        input_features = input[0]
        input_image = input[1]
        model_feat = model_features(9, 3)
        out_features_model = model_feat(input_features) 
        print('output from model_feat: ', out_features_model.shape)

        model_img = model_image(3,1)
        out_image_model = model_img(input_image)
        print('output from model_img: ', out_image_model.shape)
        fused_features = torch.concat((out_features_model, out_image_model), dim = 1)  # fusion
        model_fuse = decoder(inchannel_size= 33, outchannel_size= 1)
        output = model_fuse(fused_features)
        print('output from model_fusion: ', output, output.shape)

        return output

# test the model
input = (torch.randn(16, 9),torch.randn(16, 3, 683, 1024))
output = fusion_fun(input)



