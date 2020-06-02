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


class Gumbel_Net(nn.Module):
    # Accept x, put it into linear transformation,
    # pass it through gumbel_softmax, expand dimension into image size and output it

    def __init__(self, num_model, f_dim):
        super(Gumbel_Net, self).__init__()
        self.num_model = num_model
        self.f_dim = f_dim
        #self.x_dim = 150528
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mlp_layers = []
        mlp_layers.append(nn.Linear(self.f_dim, self.num_model))
        mlp_layers.append(nn.BatchNorm1d(self.num_model))
        mlp_layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*mlp_layers)



    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """

        y = self.gumbel_softmax_sample(logits, temperature) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y

        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """
        Draw a sample from the Gumbel-Softmax distribution
        """
        noise = self.sample_gumbel(logits)
        y = (logits + noise) / temperature
        return F.softmax(y, dim=1)

    def sample_gumbel(self, logits):
        """
        Sample from Gumbel(0, 1)
        """
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()

        return Variable(noise.float()).to(self.device)

    def forward(self, feature, temperature, hard):
        # x: # 64, 150528
        # feature: # 64, 240, 14, 14
        # out: 64 x (3072 + 2 x 47040)
        feature = feature.contiguous().view(feature.size(0), -1)
        # print(feature.shape) # 6, 47040
        out = feature

        out = self.mlp(out)

        # 1, 2
        #tmp = F.softmax(out, dim=1) 
        #print(tmp.shape) # batch, num_model
        logit = F.softmax(out, dim=1)[:1]

        out = self.gumbel_softmax(out, temperature, hard) # batch x num_generator

        gumbel_out = out.clone()
        # batch x num_model x 1
        out = out.unsqueeze(2) # batch x num_generator x 3 x imsize x imsize
        # batch x num_model x 10
        out = out.repeat(1, 1, 10)
        
        return out, gumbel_out, logit