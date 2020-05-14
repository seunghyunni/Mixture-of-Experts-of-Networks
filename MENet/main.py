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
from dataset import Dataset, create_loader


# Configuration
args = \
easydict.EasyDict({"image_size": 224,
                   "x_dim": 23520,
                   "num_model": 2,
                   "lambda_gp": 10,
                   "save_dir": "./checkpoint_0511",

                   # Training Settings
                   "epochs": 1000,
                   "batch_size": 128,
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
                    "balance_weight" : 0.1,
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

# Dataset
traindata_dir = "../ImageNet/train/ILSVRC/Data/CLS-LOC/train"
validdata_dir = "../ImageNet/train/ILSVRC/Data/CLS-LOC/val"

transform =transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])


# Train Dataset & Loader
trainset = Dataset(traindata_dir)
trainloader = create_loader(
    dataset = trainset,
    input_size=(3,224,224),
    batch_size = args.batch_size,
    interpolation= "bicubic",
    mean= (0.485, 0.456, 0.406),
    std= (0.229, 0.224, 0.225),
    num_workers=2,
    crop_pct=1.0 )
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, drop_last= True)

# Test Dataset & Loader
validset = Dataset(validdata_dir)
validloader = create_loader(
    dataset = validset,
    input_size=(3,224,224),
    batch_size = args.batch_size,
    interpolation= "bicubic",
    mean= (0.485, 0.456, 0.406),
    std= (0.229, 0.224, 0.225),
    num_workers=2,
    crop_pct=1.0 )
#validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False, num_workers=2)

# Define the model
C = Classifier(args.image_size, args.num_model)
C = nn.DataParallel(C)
C.to(device)

Gum = Gumbel_Net(args.num_model, f_dim = 47040)
Gum = nn.DataParallel(Gum)
Gum.to(device)

# Optimizer
c_optimizer = torch.optim.Adam(C.parameters(), args.c_lr, [args.beta1, args.beta2])
gum_optimizer = torch.optim.Adam(Gum.parameters(), args.g_lr, [args.beta1, args.beta2], weight_decay=0.1)

# Loss
criterion = nn.CrossEntropyLoss()

def apply_gumbel(pred, gumbel_out):
    #print("Gumbell output shape")
    #print(torch.mul(pred, gumbel_out).shape) 64, 2, 1000
    #print(torch.sum(torch.mul(pred, gumbel_out), dim=1).shape) 64, 1000
    return torch.sum(torch.mul(pred, gumbel_out), dim=1)


# def load_pretrained_model(self):
#   G.load_state_dict(torch.load(os.path.join(args.model_save_path, '{}_G.pth'))))
#   D.load_state_dict(torch.load(os.path.join(args.model_save_path, '{}_D.pth')))
#   Gum.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_Gum.pth'))))
#   print('loaded trained models (step: {})..!'))


best_loss = 100.0

# Train  
for epoch in range(args.epochs):
    print('\n===> Epoch [%d/%d]' % (epoch+1, args.epochs))

    c_running_loss = 0.0
    gum_running_loss = 0.0
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
        # print("Prediction shape")
        # print(pred.shape) # 64, 2, 1000
        # print("Feature shape")
        # print(feature.shape) # 64, 240, 14, 14
        out, gumbel_out, logit = Gum(feature, args.gum_t, True)
        output = apply_gumbel(pred, out)

        c_loss = criterion(output, labels)

        balance_weight = args.balance_weight

        target = Variable(torch.ones(args.num_model)).to(device) / args.num_model
        dist = gumbel_out.sum(dim=0) / gumbel_out.sum()
        balance_loss = F.mse_loss(dist, target) * balance_weight + c_loss

        # print("====balance Loss=======")
        # print(target) 0.5, 0.5 
        # print(dist) 0.4, 0.6
        # print(F.mse_loss(dist, target))

        c_optimizer.zero_grad()
        gum_optimizer.zero_grad()

        #c_loss.backward(retain_graph=True)
        balance_loss.backward()

        c_optimizer.step()
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

    with torch.no_grad():
        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # Forward
            pred, feature = C(inputs)
            out, gumbel_out, logit = Gum(feature, args.gum_t, True)
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
    print('     - Accuracy of the network on the 10000 test images: %d ' % (val_acc))
    
    if val_loss <= best_loss:
        best_loss = val_loss
        torch.save(C.state_dict(), os.path.join(args.save_dir, '{:03d}'.format(int(epoch+1)) +'_{:05.4f}'.format(val_loss) +'_{:05.4f}'.format(val_acc) +'.pt'))



       
