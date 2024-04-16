import torch
import torch.nn.functional as F
from unsupervised.utils import accuracy

torch.manual_seed(0)


def info_nce_loss(features, batch_size, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    return logits, labels


def train_one_epoch(train_loader, model, criterion, optimizer, args):
    top1_list = []
    top5_list = []
    acc_list = []
    loss_list = []
    for i, (images, labels) in enumerate(train_loader):
        images = torch.cat(images, dim=0)
        images = images.cuda()
        labels = torch.cat([labels for _ in range(args.n_views)], dim=0)
        labels = labels.cuda()

        features, logits = model(images)
        con_logits, con_labels = info_nce_loss(
            features, batch_size=args.batch_size, n_views=args.n_views, temperature=args.temperature
        )
        loss1 = criterion(con_logits, con_labels)
        loss2 = criterion(logits, labels)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        top1, top5 = accuracy(con_logits, con_labels, topk=(1, 5))
        acc = accuracy(logits, labels, topk=(1,))[0]
        top1_list.append(top1.item())
        top5_list.append(top5.item())
        acc_list.append(acc.item())

        if (i + 1) % args.log_steps == 0:
            print("Iter{:0>2d} Loss: {:.4f}, top1: {:.4f}, top5: {:.4f}, acc:{:.4f}".format(
                i + 1, loss.item(), top1.item(), top5.item(), acc.item()
            ))

    return top1_list, top5_list, acc_list, loss_list
