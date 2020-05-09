import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import time
import datetime
import copy
import numpy as np
import geffnet


class Classifier(nn.Module):
    def __init__(self, image_size=32, x_dim=3072, num_model=2):
        super(Classifier, self).__init__()
        self.image_size = image_size
        
        self.num_model = num_model
        self.num_class = 10

        # MixNet S
        self.mixnet_s = geffnet.create_model('mixnet_s', pretrained=False) 
        num_ftrs_1 = self.mixnet_s.classifier.in_features
        self.mixnet_s.classifier = nn.Sequential(
            nn.Linear(num_ftrs_1, self.num_class),
            nn.Softmax(dim=0))
        
        # MixNet M
        self.mixnet_m = geffnet.create_model('mixnet_m', pretrained=False)
        num_ftrs_2 = self.mixnet_m.classifier.in_features
        self.mixnet_m.classifier = nn.Sequential(
            nn.Linear(num_ftrs_2, self.num_class),
            nn.Softmax(dim=0))
        
        # Split model 
        self.all_layers1 = nn.ModuleList()
        
        # 120 channel
        self.all_layers2 = nn.ModuleList()
        
        # classifier
        self.all_layers3 = nn.ModuleList()

        self.relu = nn.ReLU(inplace=True)

        for i in range(self.num_model):
            if i == 0:
                self.main1 = nn.Sequential(self.mixnet_s.conv_stem,
                                        self.mixnet_s.bn1,
                                        self.mixnet_s.act1,
                                        self.mixnet_s.blocks[:5]
                                        )
                
                self.main2 = nn.Sequential(self.mixnet_s.blocks[5],
                                        self.mixnet_s.conv_head,
                                        self.mixnet_s.bn2,
                                        self.mixnet_s.act2,
                                        self.mixnet_s.global_pool,
                                        )
                
                self.main3 = nn.Sequential(self.mixnet_s.classifier)

            else:
                self.main1 = nn.Sequential(self.mixnet_m.conv_stem,
                                            self.mixnet_m.bn1,
                                            self.mixnet_m.act1,
                                            self.mixnet_m.blocks[:5]
                                            )

                self.main2 = nn.Sequential(self.mixnet_m.blocks[5],
                                            self.mixnet_m.conv_head,
                                            self.mixnet_m.bn2,
                                            self.mixnet_m.act2,
                                            self.mixnet_m.global_pool
                                            )

                self.main3 = nn.Sequential(self.mixnet_m.classifier)
            
            self.all_layers1.append(self.main1)
            self.all_layers2.append(self.main2)
            self.all_layers3.append(self.main3)
  
    def forward(self, x):
        
        # Feature
        # 16 x 1 x channel x feature_map_size x feature_map_size
        # 16 x 1 x 120 x 2 x 2
        feature = self.all_layers1[0](x).unsqueeze(1)
        
        # Main1
        for i in range(1, self.num_model):  # from 1 to ~
            temp = self.all_layers1[i](x).unsqueeze(1)
            feature = torch.cat([feature, temp], dim=1)
        
        # feature (by concat)
        # 16 x num_model x channel x feature_map_size x feature_map_size
        # ex) 16 x 2 x 120 x 2 x 2
        # Main2 & 3
        output = self.all_layers2[0](feature[:,0]).unsqueeze(1) # feature[:,0]= 16 x 120 x 2 x 2
        output = output.flatten(1)
        output = self.all_layers3[0](output).unsqueeze(1)
        
        for i in range(1, self.num_model):  # from 1 to ~
            temp = self.all_layers2[i](feature[:, i]).unsqueeze(1)
            temp = temp.flatten(1)
            temp = self.all_layers3[i](temp).unsqueeze(1)
            output = torch.cat([output, temp], dim=1)

        # output (by concat)
        # 16 x 2 x 10

        # batch x num_model x _dim
        # 16 x 2 x 480
        feature = feature.contiguous().view(feature.size(0), 2, -1)
        feature = self.relu(feature)
        # 16 x 960
        feature = feature.contiguous().view(feature.size(0), -1)

        return output, feature