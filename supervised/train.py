import torch

from tqdm import tqdm
from supervised.utils import accuracy


def train_one_epoch(train_loader, model, criterion, optimizer, args):
    loss_list = []
    acc_list = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        global_feature, global_logit = model(images, labels, args)
        loss = criterion(global_logit, labels)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(global_logit, labels, topk=(1,))[0]
        acc_list.append(acc.item())

        if i % args.log_steps == 0:
            print("Iter{:0>2d} Loss: {:.4f}, Acc: {:.4f}".format(i, loss.item(), acc.item()))

    return loss_list, acc_list
