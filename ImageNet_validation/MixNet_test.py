import geffnet
import ssl
import urllib
import torch
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
from matplotlib.pyplot import imsave
from matplotlib import pyplot as plt 
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import warnings
import math
import random
from data import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dataset import Dataset
from utils import accuracy, AverageMeter
from config import resolve_data_config
import torchvision
from pandas import DataFrame as df
import json


# GPU Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

loss_fn = nn.CrossEntropyLoss().cuda()

class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]

    return transforms.Compose(tfl)

def fast_collate(batch):
    targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    batch_size = len(targets)
    tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        tensor[i] += torch.from_numpy(batch[i][0])

    return tensor, targets


class PrefetchLoader:

    def __init__(self,
            loader,
            rand_erase_prob=0.,
            rand_erase_mode='const',
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        if rand_erase_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=rand_erase_prob, mode=rand_erase_mode)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x

def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        rand_erase_prob=0.,
        rand_erase_mode='const',
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
):
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    transform = transforms_imagenet_eval(
        img_size,
        interpolation=interpolation,
        use_prefetcher=use_prefetcher,
        mean=mean,
        std=std,
        crop_pct=crop_pct)

    dataset.transform = transform

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    if collate_fn is None:
        collate_fn = fast_collate 

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=is_training,
    )
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            rand_erase_prob=rand_erase_prob if is_training else 0.,
            rand_erase_mode=rand_erase_mode,
            mean=mean,
            std=std)

    return loader


#m = timm.create_model('mobilenetv2_100', pretrained=False)
m = geffnet.create_model('mnasnet_a1', pretrained = True)
m.eval()

m.to(device) 

batch_size = 50
data = "./ImageNet/val"
workers = 2
print_freq = 1
args = None

print('Model created, param count: %d' %(sum([m1.numel() for m1 in m.parameters()])))

#data_config = resolve_data_config(model, args)
#model, test_time_pool = apply_test_time_pool(model, data_config, args)

print("Loading Data")
loader = create_loader(
    Dataset(data),
    input_size=(3,224,224),
    batch_size=batch_size,
    use_prefetcher=True,
    interpolation= "bicubic",
    mean= (0.485, 0.456, 0.406),
    std= (0.229, 0.224, 0.225),
    num_workers=workers,
    crop_pct=1.0 )

print("Loaded %d images" %len(loader))

batch_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

end = time.time()

result = dict()

df_label = [] 
df_time = [] 
df_loss = []
df_prec1 = []
df_prec5 = []
df_confidence = []

with torch.no_grad():
    for i, (input, target) in enumerate(loader):
        target = target.cuda()
        input = input.cuda()
        
        #tmp = dict()
        
        # compute output
        output = m(input)
        output_prob = torch.nn.functional.softmax(output, dim =1)
        loss = loss_fn(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        #print(output.data.topk(1, 1, True, True)[0])
        confidence = output_prob.data.topk(1, 1, True, True)[0]
        confidence = confidence.t()
        confidence = confidence.tolist()[0]
        confidence = round(sum(confidence)/len(confidence), 4)
        
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Label: [{label[0]}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s)  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}),  '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(loader), label = target, batch_time=batch_time, 
                rate_avg=input.size(0) / batch_time.avg,
                loss=losses, top1=top1, top5=top5))
        
        # # json file
        # tmp['time'] = round(input.size(0) / batch_time.avg, 3)
        # tmp['loss'] = round(losses.avg,4)
        # tmp['top1'] = round(top1.avg, 4)
        # tmp['top5'] = round(top5.avg, 4)
        # tmp['confidence'] = confidence
        
        # csv columns
        df_label.append(target.data[0].item())
        df_time.append(round(input.size(0) / batch_time.avg, 3))
        df_loss.append(round(losses.avg,4))
        df_prec1.append(round(top1.avg, 4))
        df_prec5.append(round(top5.avg, 4))
        df_confidence.append(confidence)
        
        #result[str(target.data[0].item())] = tmp

print(' * Prec@1 {top1.avg:.3f} ({top1a:.3f}) Prec@5 {top5.avg:.3f} ({top5a:.3f})'.format(
    top1=top1, top1a=100-top1.avg, top5=top5, top5a=100.-top5.avg))

# To Json
# with open ('./minetM_train_set.json', 'w', encoding = 'utf-8') as make_file:
#     json.dump(result, make_file, indent = "\t")

# print("Completed saving Json file")
# To Dataframe 
output_df = df(np.array([df_label, df_time, df_loss, df_prec1, df_prec5, df_confidence]).T, columns=['label', 'rating_time(/s)', 'LossAvg', 'Prec1(%)', 'Prec5(%)', 'confidence'])
output_df.to_csv("./mnasnet_a1_result.csv")

print("Completed saving csv file")