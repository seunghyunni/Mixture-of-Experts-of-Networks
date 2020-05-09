import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from utils import AccumulatedAccuracyMetric
from model import FinetuneResnet
from train import fit
from tensorboardX import SummaryWriter
import geffnet


# Define Parameters
class config():s
    # Root directory for dataset
    traindata_dir = "../cifar10"
    testdata_dir = "../cifar10"

    save_dir = "./checkpoint/"

    # Number of class
    n_class = 10

    # Name of class
    class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Number of workers for dataloader
    workers = 2
    batch_size = 32
    image_size = 32
    num_epochs = 20
    lr = 1e-3

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


# Gets the name of a device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()

# Model
# Embedding_Net1 (Main Model)
#embedding_net_1 = models.resnet50(pretrained=True)
embedding_net_1 = geffnet.create_model('mobilenetv3_large_100', pretrained=False) 

num_ftrs_1 = embedding_net_1.classifier.in_features
embedding_net_1.classifier = nn.Sequential(
    nn.Linear(num_ftrs_1, config.n_class),
    nn.Softmax(dim=0))


# Embedding_Net2 (for training smoothing parameter)
# embedding_net_2 = FinetuneResnet(10)

# Entire Network (Embedding_Net1 + Embedding_Net2)
model= embedding_net_1
model = nn.DataParallel(model)
model.to(device)

model2 = FinetuneResnet(10)
model2 = nn.DataParallel(model2)
model2.to(device)

# load pretrained weights if possible
pkl_path = None

try:
    model.load_state_dict(torch.load(pkl_path, map_location=device))
    model2.load_state_dict(torch.load(pkl_path, map_location=device))
    print("\n--------model restored--------\n")
    print("\n--------model2 restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# loss function
#loss_fn = MOD_CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss

# parameters
lr = config.lr
optimizer = optim.Adam(model.parameters(), betas=[.9, .999], weight_decay=0.0, lr=lr)
optimizer_smoothing = optim.Adam(model2.parameters(), betas=[.9, .999], weight_decay=0.0, lr=lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = config.num_epochs
log_interval = 100


# DataLoader
# Train Dataset
trainset = torchvision.datasets.CIFAR10(root='../cifar10', train = True, download=False, transform = config.transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last= True)

# Validation Dataset
validset = torchvision.datasets.CIFAR10(root='../cifar10', train = False, download=False, transform = config.transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=config.batch_size, shuffle=False, num_workers=2)

# Tensorboard
train_writer = SummaryWriter('./checkpoint/logs/')

# Train, Validate
fit(config.save_dir, train_writer, trainloader, validloader, model, model2, loss_fn, optimizer, optimizer_smoothing, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
