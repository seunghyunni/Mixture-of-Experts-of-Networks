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
from model import Classifier
from utils import Gumbel_Net
import easydict


# Configuration
args = \
easydict.EasyDict({"image_size": 32,
                   "x_dim": torch.randn(3, 32, 32).view(-1).size(0),
                   "num_model": 2,
                   "lambda_gp": 10,
                   "save_dir": "./checkpoint",

                   # Training Settings
                   "epochs": 1000,
                   "batch_size": 32,
                   "c_lr" : 0.0002,
                   "g_lr" : 0.0002,
                   "beta1": 0.0,
                   "beta2": 0.9,                   
                   "use_tensorboard": False,

                   # Gumbel hyperparameters
                   "gum_t" : 1, 
                   "gum_orig": 1,
                   "gum_t_decay": 0.0001,
                   "step_t_decay": 1,
                   "start_anneal": 0,
                   "min_t": 0.01,

                   # Step size
                    "log_step" : 10,
                    "sample_step" : 100,
                    "model_save_step" : 780,
                    "score_epoch" : 5,
                    "score_start" : 5,

                    #load-balancing
                    "load_balance" : True,
                    "balance_weight" : 1.0,
                    "matching_weight" : 1.0})


# GPU Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("cuda")
else:
    print("CPU")

os.makedirs(args.save_dir, exist_ok=True)

transform =transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

# Train Dataset & Loader
trainset = torchvision.datasets.CIFAR10(root='../cifar10', train = True, download=False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, drop_last= True)

# Test Dataset & Loader
testset = torchvision.datasets.CIFAR10(root='../cifar10', train = False, download=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Define the model
C = Classifier(args.image_size, args.x_dim, args.num_model)
C = nn.DataParallel(C)
C.to(device)

Gum = Gumbel_Net(args.num_model, args.x_dim)
Gum = nn.DataParallel(Gum)
Gum.to(device)

# Optimizer
c_optimizer = torch.optim.Adam(C.parameters(), args.c_lr, [args.beta1, args.beta2])
gum_optimizer = torch.optim.Adam(Gum.parameters(), args.g_lr, [args.beta1, args.beta2], weight_decay=0.1)

# Loss
criterion = nn.CrossEntropyLoss()

def apply_gumbel(pred, gumbel_out):
    return torch.sum(torch.mul(pred, gumbel_out), dim=1)


# def load_pretrained_model(self):
#   G.load_state_dict(torch.load(os.path.join(args.model_save_path, '{}_G.pth'))))
#   D.load_state_dict(torch.load(os.path.join(args.model_save_path, '{}_D.pth')))
#   Gum.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_Gum.pth'))))
#   print('loaded trained models (step: {})..!'))


# Train  
for epoch in range(args.epochs):
    print('\n===> Epoch [%d/%d]' % (epoch+1, args.epochs))

    c_running_loss = 0.0
    gum_running_loss = 0.0
    best_loss = 100.0
    n_samples = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        C.train()
        Gum.train()

        # ================== Train C ================== #
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        # Forward
        pred, feature = C(inputs)
        out, gumbel_out, logit = Gum(inputs.flatten(1), feature, args.gum_t, True)
        output = apply_gumbel(pred, out)

        c_loss = criterion(output, labels)

        c_optimizer.zero_grad()
        gum_optimizer.zero_grad()

        c_loss.backward(retain_graph=True)

        c_optimizer.step()
        gum_optimizer.step()

        # ===================Train Gumbel ===================#
        if args.load_balance == True:
            balance_weight = args.balance_weight

            pred, feature = C(inputs)
            out, gumbel_out, logit = Gum(inputs.flatten(1), feature, args.gum_t, True)

            target = Variable(torch.ones(args.num_model)).to(device) / args.num_model
            dist = gumbel_out.sum(dim=0) / gumbel_out.sum()
            balance_loss = F.mse_loss(dist, target) * balance_weight

            c_optimizer.zero_grad()
            gum_optimizer.zero_grad()

            balance_loss.backward()
            
            gum_optimizer.step()

        # Print statistics
        c_running_loss += c_loss.item()
        gum_running_loss += balance_loss.item()

        # Accuracy
        _, predicted = torch.max(output.data, 1)
        n_batch = int(inputs.size()[0])
        n_samples += n_batch
        correct += (predicted == labels).sum().item()

        # Print every 100 mini-batches
        if i % 200 == 199:
            if args.load_balance == True:
                # elapsed = time.time() - start_time
                # elapsed = str(datetime.timedelta(seconds=elapsed))
                print('     - Iteration [%5d / %5d] --- Classification_Loss: %.3f     Gating_Loss: %.3f' % (i+1, len(trainloader), c_running_loss / 200, gum_running_loss / 200))
                print("         - Gumbel choice for 1 instances : ", gumbel_out.max(dim=1)[1].data[0])
                print("         - Logit choice (underlying distribution :", logit.data[0])
                print('         - Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / n_samples))
                c_running_loss = 0.0
                gum_running_loss = 0.0
                correct = 0
                n_samples = 0

    
    # Validation
    C.eval()
    Gum.eval()
    correct = 0
    n_samples = 0
    val_loss = 0
    val_acc = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        # Forward
        pred, feature = C(inputs)
        out, gumbel_out, logit = Gum(inputs.flatten(1), feature, args.gum_t, True)
        output = apply_gumbel(pred, out)

        n_batch = int(inputs.size()[0])
        n_samples += n_batch
        val_loss += criterion(output, labels).item()

        # Accuracy
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()

    val_loss = val_loss / n_samples
    val_acc = 100 * correct / n_samples
    print("===========================================================================")
    print('     - Validation: Classification loss %.4f \n' % (val_loss))
    print('     - Accuracy of the network on the 10000 test images: %d %%' % (val_acc))
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(C.state_dict(), os.path.join(args.save_dir, '{:03d}'.format(int(epoch+1)) +'_{:05.4f}'.format(val_loss)+'.pt'))



       
