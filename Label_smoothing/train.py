import torch
import numpy as np
from utils import show_plot
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from utils import smooth

def fit(save_dir, train_writer, train_loader, val_loader, model, model2, loss_fn, optimizer, optimizer_smoothing, scheduler, n_epochs, cuda, log_interval, metrics=[], start_epoch=0):
    best_loss = 0.0  # Only Save Results when Score is best.
    counter = []
    loss_history = []

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train Stage
        train_loss, metrics = train_epoch(train_loader, model, model2, loss_fn, optimizer, optimizer_smoothing, cuda, log_interval, metrics)

        counter.append(train_loss)
        loss_history.append(epoch + 1)

        # Write to Tensorboard
        train_writer.add_scalar("train/loss", train_loss, epoch + 1)
        train_writer.add_scalar("train/acc", metrics[0].value() , epoch + 1)

        # print
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        # Validation Stage
        val_loss, metrics = valid_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        # Write to Tensorboard
        train_writer.add_scalar("valid/loss", val_loss, epoch + 1)
        train_writer.add_scalar("valid/acc", metrics[0].value(), epoch + 1)

        # print
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

        # Only save results when best validation loss updates
        if val_loss > best_loss:
            best_loss = val_loss
            # save results
            pkl_save_dir = os.path.join(save_dir, 'pkl')
            if not os.path.exists(pkl_save_dir):
                os.makedirs(pkl_save_dir)

            torch.save(model.state_dict(), os.path.join(save_dir, 'pkl', 'epoch_%d_loss_%.4f_accuracy_%.4f.pth' % (
                (epoch + 1), val_loss, metrics[0].value())))

    show_plot(counter, loss_history)


def train_epoch(train_loader, model, model2, loss_fn, optimizer, optimizer_smoothing, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()
    model.train()
    model2.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data).cuda().float()
        target = Variable(target).cuda().float()

        # Initialize optimizer
        optimizer.zero_grad()
        optimizer_smoothing.zero_grad()

        # label smoothing
        smoothing = model2(data, target)
        #smoothed_target = torch.empty(size=(target.size(0), 10),device=target.device).fill_(smoothing[:,0][0] /(10-1)).scatter_(1, target.data.unsqueeze(1), 1.-smoothing)
        smoothed_target = smooth(target, smoothing)

        # Predict
        outputs = model(data)

        # Loss
        loss = loss_fn(outputs, smoothed_target)
        losses.append(loss.item())
        total_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer_smoothing.zero_grad()

        # print
        for metric in metrics:
            metric(outputs, target)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data[0]), len(train_loader.dataset),
                                                                      100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def valid_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        losses = []
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data = Variable(data).cuda().float()
            target = Variable(target).cuda().float()

            # predict
            outputs = model(data)

            # loss
            loss = loss_fn(outputs, target)
            losses.append(loss.item())
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target)

    return val_loss, metrics


