import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from utils import AccumulatedAccuracyMetric, MOD_CrossEntropyLoss
from model import Entire_Net, FinetuneResnet
import os
from torch.autograd import Variable
import numpy as np

testdata_dir = "/content/gdrive/My Drive/SKT/data/test/"
save_dir = "/content/gdrive/My Drive/SKT/save/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Number of workers for dataloader
workers = 2
batch_size = 8
image_size = 32
num_epochs = 20
lr = 1e-3

# Normalize
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

# DataLoader
test_set = torchvision.datasets.CIFAR10(root='./data', train = True, download=True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True, num_workers=2)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Gets the name of a device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()

# Model
# Embedding_Net1 (Main Model)
embedding_net_1 = models.resnet50(pretrained=True)
num_ftrs_1 = embedding_net_1.fc.in_features # 2048

embedding_net_1.fc = nn.Sequential(
    nn.Linear(num_ftrs_1, 1024),
    nn.ReLU(),
    nn.Linear(1024, 516),
    nn.ReLU(),
    nn.Linear(516, 10),
    nn.Softmax())

# Embedding_Net2 (for training smoothing parameter)
embedding_net_2 = FinetuneResnet(10)

# Entire Network (Embedding_Net1 + Embedding_Net2)
model = Entire_Net(embedding_net_1, embedding_net_2)
model.to(device)

# load pretrained weights if possible
pkl_path = None

try:
    model.load_state_dict(torch.load(pkl_path, map_location=device))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# loss function
loss_fn = MOD_CrossEntropyLoss()

# parameters
log_interval = 1

metrics = []
# Inference
if __name__ == "__main__":
    metrics = []
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        losses = []
        model.eval()
        loss = 0
        test_loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data = Variable(data).cuda().float()
            target = Variable(target).cuda().float()

            # predict
            outputs = model(data)

            # loss
            loss_inputs = outputs
            target = (target,)
            loss_inputs += target

            loss_outputs = loss_fn(loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            losses.append([loss.item()])
            test_loss += loss.item()

            # print
            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                message = 'Validation: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data[0]),
                                                                          len(test_loader.dataset),
                                                                          100. * batch_idx / len(test_loader),
                                                                          np.mean(losses))
                print(message)

        test_loss /= len(test_loader)
        mean = test_loss
        std = torch.std(torch.stack(losses))

        # print
        print("================ Finished ================")
        print("Mean, std of test set: %.4f , %.4f" % (mean, std))

