import os
import sys
import argparse
import datetime
import warnings
import statistics as st

import torch
from torch.utils.data import DataLoader

from supervised.dataset import Dataset, Dataset1K
from supervised.model import Model, DOLG, TransformerModel, Model101, TransformerLargeModel
from supervised.train import train_one_epoch

parser = argparse.ArgumentParser()
# for dataset
parser.add_argument('--split', default='./dataset/new_dataset/train_test_split.pk')
# for model
parser.add_argument('--backbone', default="vit", type=str)
parser.add_argument('--num_class', default=1000, type=int)
parser.add_argument('--softmax', default=1, type=int)
parser.add_argument('--m', default=0.15, type=float)
parser.add_argument('--s', default=30, type=float)
# for training
parser.add_argument('--max_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr_base', default=1e-4, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--pretrain_weight', default=None, type=str)
# for visualization
parser.add_argument('--log_steps', default=50, type=int)
parser.add_argument('--save_steps', default=5, type=int)

if __name__ == "__main__":
    warnings.filterwarnings("error")

    args = parser.parse_args()

    save_dir = "ckpt/wrinkle_full/{}_lr{}_bs{}_epoch{}_softmax".format(
        args.backbone, args.lr, args.batch_size, args.max_epochs
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set up dataset
    dataset = Dataset1K(args.split)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

    if args.backbone == "resnet":
        model = Model(num_class=args.num_class, args=args).cuda()
    elif args.backbone == "dolg":
        model = DOLG(num_class=args.num_class, args=args).cuda()
    elif args.backbone == "vit":
        model = TransformerModel(num_class=args.num_class, args=args).cuda()
    elif args.backbone == "vit_large":
        model = TransformerLargeModel(num_class=args.num_class, args=args).cuda()
    elif args.backbone == "resnet101":
        model = Model101(num_class=args.num_class, args=args).cuda()
    else:
        raise NotImplementedError

    if args.pretrain_weight is not None:
        pretrain_weight = torch.load(args.pretrain_weight)
        model_dict = model.state_dict()
        missing_keys, unexpected_keys = model.load_state_dict(pretrain_weight, strict=False)
        print("construct model total {} keys and {} keys loaded".format(len(model_dict),
                                                                        len(model_dict) - len(missing_keys)))
        print("missing_keys:{}, unexpected_keys:{}".format(len(missing_keys), len(unexpected_keys)))
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0, last_epoch=-1)

    for epoch in range(args.max_epochs):
        print("************** epoch{:0>3d} *****************".format(epoch))
        loss_list, acc_list = train_one_epoch(train_loader, model, criterion, optimizer, args)
        print("Epoch {:0>2d}, mean loss:{:.4f}, mean acc:{:.4f} lr:{}".format(epoch, st.mean(loss_list),
                                                                              st.mean(acc_list),
                                                                              scheduler.get_last_lr()))
        print("time:", datetime.datetime.now().time())
        scheduler.step()
        if (epoch + 1) % args.save_steps == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_{:0>2d}.ckpt".format(epoch)))
            print("save model to:", os.path.join(save_dir, "model_{:0>2d}.ckpt".format(epoch)))
