import os
import sys
import argparse
import datetime
import statistics as st

import torch
from torch.utils.data import DataLoader

from unsupervised.dataset import Dataset
from unsupervised.model import ResNetSimCLR, VitSimCLR
from unsupervised.train import train_one_epoch

parser = argparse.ArgumentParser()
# for dataset
parser.add_argument('--data', default='./dataset/new_dataset')
parser.add_argument('--train_list', default='./dataset/new_dataset/train.txt')
parser.add_argument('--n_views', default=2)
# for model
parser.add_argument('--backbone', default="vit", type=str)
parser.add_argument('--num_class', default=1000, type=int)
parser.add_argument('--out_dim', default=2048, type=int)
parser.add_argument('--temperature', default=0.07)
# for training
parser.add_argument('--max_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--pretrain_weight', default=None, type=str)
# for visualization
parser.add_argument('--log_steps', default=20, type=int)
parser.add_argument('--save_steps', default=5, type=int)

if __name__ == "__main__":
    args = parser.parse_args()

    save_dir = "ckpt/wrinkle_certify/{}_lr{}_bs{}_epoch{}_translate".format(
        args.backbone, args.lr, args.batch_size, args.max_epochs
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set up dataset
    dataset = Dataset(args.data, args.train_list, n_views=args.n_views)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    if args.backbone == "resnet":
        model = ResNetSimCLR(out_dim=args.out_dim).cuda()
    elif args.backbone == "vit":
        model = VitSimCLR(out_dim=args.out_dim, num_class=args.num_class).cuda()
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0, last_epoch=-1)

    for epoch in range(args.max_epochs):
        print("************** epoch{:0>3d} *****************".format(epoch))
        top1_list, top5_list, acc_list, loss_list = train_one_epoch(train_loader, model, criterion, optimizer, args)
        print("Epoch {:0>2d}, mean loss:{:.4f}, mean top1:{:.4f} mean top5:{:.4f}, mean acc:{:.4f} lr:{}".format(
            epoch, st.mean(loss_list), st.mean(top1_list), st.mean(top5_list), st.mean(acc_list),
            scheduler.get_last_lr())
        )
        print("time:", datetime.datetime.now().time())
        scheduler.step()
        if (epoch + 1) % args.save_steps == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_{:0>2d}.ckpt".format(epoch)))
            print("save model to:", os.path.join(save_dir, "model_{:0>2d}.ckpt".format(epoch)))
