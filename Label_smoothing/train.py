import torch
import numpy as np
from utils import show_plot, smooth
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable


def fit(save_dir, train_writer, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=0):
    best_loss = 100000000.0  # Only Save Results when Score is best.
    counter = []
    loss_history = []

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train Stage
        train_loss, acc = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval)

        counter.append(train_loss)
        loss_history.append(epoch + 1)

        # Write to Tensorboard
        train_writer.add_scalar("train/loss", train_loss, epoch + 1)
        train_writer.add_scalar("train/acc",acc , epoch + 1)

        # print
        message = 'Epoch: {}/{}. Train set - Average loss: {:.4f},  Average Accuracy:{:.2f}%'.format(epoch + 1, n_epochs, train_loss, acc)

        # Validation Stage
        val_loss, acc = valid_epoch(val_loader, model, loss_fn, cuda)
        val_loss /= len(val_loader)

        # Write to Tensorboard
        train_writer.add_scalar("valid/loss", val_loss, epoch + 1)
        train_writer.add_scalar("valid/acc", acc, epoch + 1)

        # print
        message = ""
        message += '\nEpoch: {}/{}. Validation set - Average loss: {:.4f},  Average Accuracy: {:.2f}%'.format(epoch + 1, n_epochs, val_loss, acc)


        # Only save results when best validation loss updates
        if val_loss <= best_loss:
            best_loss = val_loss
            # save results
            pkl_save_dir = os.path.join(save_dir, 'pkl')
            if not os.path.exists(pkl_save_dir):
                os.makedirs(pkl_save_dir)

            torch.save(model.state_dict(), os.path.join(save_dir, 'pkl', 'epoch_%d_loss_%.4f_accuracy_%.2f.pth' % (
                (epoch + 1), val_loss, acc)))

    # show_plot(counter, loss_history)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval):
    model.train()
    #model2.train()
    losses = []
    total_loss = 0
    correct = 0
    n_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data).cuda().float()
        target = Variable(target).cuda().float()

        # Initialize optimizer
        #optimizer_smoothing.zero_grad()

        # label smoothing
        #smoothing = model2(data, target)
        outputs, smoothing = model(data, target)
        
        summ = smoothing.sum(dim = 0)[0]
        avg = summ / smoothing.size(0)
        
        if avg > 0.9:
            avg = 0.9
        
        # smoothed_target = smooth(target, smoothing)
        # smoothed_target = Variable(smoothed_target, requires_grad = False)
        #smoothed_target = torch.empty(size=(target.size(0), 10),device=target.device).fill_(smoothing[:,0][0] /(10-1)).scatter_(1, target.data.unsqueeze(1), 1.-smoothing)
        
        # Predict
        #outputs = model(data) # batch, num_class
        
        # Loss
        # print(smoothing.shape) # batch, 1  
        # print(smoothing[0])
        loss = loss_fn(outputs, target, smoothing)
        losses.append(loss.item())
        total_loss += loss.item()

        # Accuracy
        n_batch = int(data.size()[0])
        n_samples += n_batch

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.long() == target.long()).sum().item()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #optimizer_smoothing.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)],   Loss: {:.6f},   Acc: {:.2f}%,   Smoothing: {:.4f}'.format(batch_idx, len(train_loader),
                                                                      100. * batch_idx / len(train_loader), np.mean(losses), 100 * correct / n_samples,  avg)
            # for metric in metrics:
            #     message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (n_samples + 1)
    acc = 100 * correct / n_samples
    return total_loss, acc


def valid_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        losses = []
        model.eval()
        val_loss = 0
        n_samples = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data = Variable(data).cuda().float()
            target = Variable(target).cuda().float()

            # predict
            outputs, smoothing = model(data, target)

            # loss
            loss = loss_fn(outputs, target, smoothing)
            losses.append(loss.item())
            val_loss += loss.item()

            # Accuracy
            n_batch = int(data.size()[0])
            n_samples += n_batch

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.long() == target.long()).sum().item()

    val_loss /= n_samples
    acc = 100 * correct / n_samples

    return val_loss, acc


