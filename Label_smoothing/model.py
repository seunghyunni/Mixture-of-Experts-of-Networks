import torch
import torchvision.models as models
import torch.nn as nn
import geffnet 


# Embedding Net2
class FinetuneResnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FinetuneResnet, self).__init__()

        #self.model = models.resnet18(pretrained=False)
        #self.fc1 = nn.Linear(512 + 1, 1)  # hidden vector + one-hot vector
        self.model = geffnet.create_model('mobilenetv3_large_100', pretrained=False) 
        self. num_ftrs_1 = self.model.classifier.in_features
        self.classifier = nn.Linear(self.num_ftrs_1 + 1, 1)  # hidden vector + one-hot vector
        self.n_classes = num_classes

    def forward(self, x, label):
        # Basic
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.blocks(x)

        x = self.model.global_pool(x)
        x = self.model.conv_head(x)
        x = self.model.act2(x)

        # Modify
        x = x.view(x.size(0), -1) 
        # Concatenate
        label = label.unsqueeze(-1) 
        x = torch.cat((x, label), dim = -1) 
        x = self.classifier(x) # batch, 1
        x = nn.Sigmoid()(x)

        return x


# Entire Net
class Entire_Net(nn.Module):
    def __init__(self, embedding_net_1, embedding_net_2):
        super(Entire_Net, self).__init__()
        self.embedding_net_1 = embedding_net_1
        self.embedding_net_2 = embedding_net_2

    def forward(self, x, labels):
        output1 = self.embedding_net_1(x)
        smoothing = self.embedding_net_2(x, labels)
        return output1, smoothing