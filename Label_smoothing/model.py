import torch
import torchvision.models as models
import torch.nn as nn

# Embedding Net2
class FinetuneResnet(nn.Module):
    def __init__(self, num_classes=10):
        super(FinetuneResnet, self).__init__()

        self.model = models.resnet18(pretrained=False)
        self.fc1 = nn.Linear(512 + 1, 1)  # hidden vector + one-hot vector
        self.n_classes = num_classes

    def forward(self, x, label):
        # Basic
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        # Modify
        x = x.view(x.size(0), -1) # 8 512
        # Concatenate
        label = label.unsqueeze(-1) # 8 1
        x = torch.cat((x, label), dim = -1) # 8, 513
        x = self.fc1(x) 
        print(x.shape)
        return x


# # Entire Net
# class Entire_Net(nn.Module):
#     def __init__(self, embedding_net_1, embedding_net_2):
#         super(Entire_Net, self).__init__()
#         self.embedding_net_1 = embedding_net_1
#         self.embedding_net_2 = embedding_net_2

#     def forward(self, x, labels):
#         output1 = self.embedding_net_1(x)
#         smoothing = self.embedding_net_2(x, labels)
#         return output1, smoothing