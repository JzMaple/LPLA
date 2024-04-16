import os
import sys
import argparse
import datetime
import statistics as st

import pygmtools
import torch
from torch.utils.data import DataLoader

from match.dataset import Dataset
from match.model import MatchModel
from match.train import train_one_epoch
from match.utils import PermutationLoss

parser = argparse.ArgumentParser()
# for dataset
parser.add_argument('--data', default='./dataset/dataset_v2')
parser.add_argument('--train_list', default='./dataset/dataset_v2/train.txt')
# for model
parser.add_argument('--backbone', default="vit", type=str)
parser.add_argument('--num_class', default=181, type=int)
parser.add_argument('--local_cnn_dim', default=2048, type=int)
parser.add_argument('--local_trans_dim', default=128, type=int)
parser.add_argument('--pretrain_weight', default=None, type=str)
# for training
parser.add_argument('--max_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr_backbone', default=1e-3, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
# for visualization
parser.add_argument('--log_steps', default=20, type=int)
parser.add_argument('--save_steps', default=5, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    pygmtools.BACKEND = "pytorch"

    save_dir = "output/match/{}_lr{}_bs{}_epoch{}_translate".format(
        args.backbone, args.lr, args.batch_size, args.max_epochs
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # dataset
    dataset = Dataset(args.data, args.train_list)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    # model
    model = MatchModel(num_class=args.num_class, args=args).cuda()
    if args.pretrain_weight is not None:
        pretrain_weight = torch.load(args.pretrain_weight)
        model_dict = model.state_dict()
        missing_keys, unexpected_keys = model.load_state_dict(pretrain_weight, strict=False)
        print("construct model total {} keys and {} keys loaded".format(len(model_dict),
                                                                        len(model_dict) - len(missing_keys)))
        print("missing_keys:{}, unexpected_keys:{}".format(len(missing_keys), len(unexpected_keys)))
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)
    # criterion
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()
    mat_criterion = PermutationLoss().cuda()
    criterion = [cls_criterion, mat_criterion]
    # optimizer
    if args.lr_backbone == args.lr:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        id_backbone = [id(param) for param in model.backbone.parameters()]
        other_param = [param for param in model.parameters() if id(param) not in id_backbone]
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": other_param, "lr": args.lr}
        ], args.lr, weight_decay = args.weight_decay)
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0,
                                                           last_epoch=-1)

    for epoch in range(args.max_epochs):
        print("************** epoch{:0>3d} *****************".format(epoch))
        cls_loss_list, cls_acc_list, mat_loss_list, mat_acc_list = train_one_epoch(
            train_loader, model, criterion, optimizer, args
        )
        print("Epoch {:0>2d}, cls loss:{:.4f}, cls acc:{:.4f}, mat loss :{:.4f}, mat acc:{:.4f} lr:{}".format(
            epoch, st.mean(cls_loss_list), st.mean(cls_acc_list), st.mean(mat_loss_list), st.mean(mat_acc_list),
            scheduler.get_last_lr())
        )
        print("time:", datetime.datetime.now().time())
        scheduler.step()
        if (epoch + 1) % args.save_steps == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_{:0>2d}.ckpt".format(epoch)))
            print("save model to:", os.path.join(save_dir, "model_{:0>2d}.ckpt".format(epoch)))
