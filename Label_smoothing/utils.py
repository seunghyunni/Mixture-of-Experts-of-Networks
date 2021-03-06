import torch
import os
import shutil
import glob
import csv
import operator
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import torch.nn as nn
import copy


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes = 1000, dim=-1, batch_size = 4):
        super(LabelSmoothingLoss, self).__init__()
        self.cls = classes
        self.dim = dim
        self.batch = batch_size

    def forward(self, pred, target, smoothing):
        # pred = pred.log_softmax(dim=self.dim)
        #confidence = 1.0 - smoothing
        with torch.no_grad():
            # true_dist = pred.data.clone()
            s = 0
            true_dist = torch.zeros_like(pred)
            # print(true_dist.shape) # b, 1000
            # print(smoothing.shape) # batch, 1
            for i in range(pred.size(0)):
                if smoothing[i][0] < 0.1:
                    s = 0.1
                else:
                    s = smoothing[i][0]
                true_dist[i].fill_(s / (self.cls - 1))
                true_dist[i].scatter_(0, target.data.long().unsqueeze(1)[i], 1.0 - s)
        
        return torch.mean(torch.sum(-true_dist * pred))


def imshow(img):
  # Unnormalize
  img = img/2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def show_plot(iteration,loss):
  plt.plot(iteration,loss)
  plt.show()


# class MOD_CrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(MOD_CrossEntropyLoss, self).__init__()
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, output1, output2, target_one_hot):

#         # with torch.no_grad():
#         tmp_result = []

#         # label = target.clone().detach()
#         # label = copy.deepcopy(target)
#         # label = torch.empty_like(output1).fill_(0).scatter_(1, label.unsqueeze(1), 1).detach()

#         # Batch_iteration ex) 1 ~ 8
#         for idx in range(32):
#             # Class_iteration ex) 1 ~ 10
#             tmp_tensor = None
#             for i, value in enumerate(output1[idx]):
#                 tmp_tensor = torch.zeros(output1[idx].shape)
#                 # max
#                 if 1 == target_one_hot[idx][i]:
#                     tmp_tensor[i] = min(value, output2[idx][i])
#                 # min
#                 else:
#                     tmp_tensor[i] = max(value, output2[idx][i])

#             tmp_result.append(tmp_tensor.cuda())

#         result = tmp_result[0]
#         for val in tmp_result[1:]:
#             result = torch.cat((result, val), dim=0)

#         output1 = result.reshape(8, 10).cuda()

#         logs = self.log_softmax(output1)
#         loss = -torch.sum(logs * target_one_hot, dim=1)

#         return loss.sum() / 32


def smooth(label, smoothing, n_class = 1000, batch_size = 64):
    result = None

    # with torch.no_grad():
    class_num = n_class

    delta = 0.9
    for idx, weight in enumerate(smoothing):
        if weight > delta:
            smoothing[idx] = 0.9
        elif weight <= 0:
            smoothing[idx] = 0

    outputs = []

    # Front
    Front = torch.ones((batch_size, 1)).cuda()
    Behind = (1 / class_num) * torch.ones((batch_size, class_num)).cuda()
    Behind = Behind.cuda()

    for idx, weigth in enumerate(smoothing):
        # 1 - w
        f = (Front[idx] - weigth) * label[idx]
        # w
        b = weigth * Behind[idx]
        outputs.append(f + b)

    result = outputs[0]

    for val in outputs[1:]:
        result = torch.cat((result, val), dim=0)

    result = result.reshape(batch_size, n_class).cuda()

    if result is None:
        pass

    return result


class CheckpointSaver:
    def __init__(
            self,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            verbose=True,
            max_history=10):

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.verbose = verbose
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, state, epoch, metric=None):
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if len(self.checkpoint_files) < self.max_history or self.cmp(metric, worst_file[1]):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            if metric is not None:
                state['metric'] = metric
            torch.save(state, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            if self.verbose:
                print("Current checkpoints:")
                for c in self.checkpoint_files:
                    print(c)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                shutil.copyfile(save_path, os.path.join(self.checkpoint_dir, 'model_best' + self.extension))

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                if self.verbose:
                    print('Cleaning checkpoint: ', d)
                os.remove(d[0])
            except Exception as e:
                print('Exception (%s) while deleting checkpoint' % str(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, state, epoch, batch_idx):
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        torch.save(state, save_path)
        if os.path.exists(self.last_recovery_file):
            try:
                if self.verbose:
                    print('Cleaning recovery', self.last_recovery_file)
                os.remove(self.last_recovery_file)
            except Exception as e:
                print("Exception (%s) while removing %s" % (str(e), self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
