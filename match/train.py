import torch

from supervised.utils import accuracy
from match.utils import matching_recall


def train_one_epoch(train_loader, model, criterion, optimizer, args):
    cls_loss_list = []
    cls_acc_list = []
    mat_loss_list = []
    mat_acc_list = []
    for i, (img1, img2, img3, M, labels) in enumerate(train_loader):
        img1 = img1.cuda()
        img2 = img2.cuda()
        M = M.cuda()
        labels = labels.cuda()

        # global_logit, pos_mat, pos_mat_dis, pos_gt_mat = model(img1, img2, M, labels)
        pos_mat, pos_mat_dis, pos_gt_mat = model(img1, img2, M, labels)

        # cls_loss = criterion[0](global_logit, labels)
        ns = nt = torch.ones((pos_mat.shape[0])).long().cuda() * pos_mat_dis.shape[1]
        mat_loss = criterion[1](pos_mat, pos_gt_mat, ns, nt)
        # cls_loss_list.append(cls_loss.item())
        mat_loss_list.append(mat_loss.item())

        # loss = cls_loss + mat_loss
        loss = mat_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # cls_acc = accuracy(global_logit, labels, topk=(1,))[0]
        # cls_acc_list.append(cls_acc.item())
        mat_acc = torch.mean(matching_recall(pos_mat_dis, pos_gt_mat, ns))
        mat_acc_list.append(mat_acc.item())

        if i % args.log_steps == 0:
            # print("Iter{:0>2d} cls loss: {:.4f}, cls acc: {:.4f}, mat loss: {:.4f}, mat acc: {:.4f}".format(
            #     i, cls_loss.item(), cls_acc.item(), mat_loss.item(), mat_acc.item()
            # ))
            print("Iter{:0>2d} cls loss: {:.4f}, cls acc: {:.4f}, mat loss: {:.4f}, mat acc: {:.4f}".format(
                i, 0, 0, mat_loss.item(), mat_acc.item()
            ))

    # return cls_loss_list, cls_acc_list, mat_loss_list, mat_acc_list
    return [0], [0], mat_loss_list, mat_acc_list
