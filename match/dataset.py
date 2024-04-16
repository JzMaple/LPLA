import copy
import sys

sys.path.append("./")

import cv2
import random
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt

from match.perspective_transforms import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list_pth, mode="train", visual=False):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.data_list_pth = data_list_pth
        self.mode = mode
        self.visual = visual

        # read data_list
        self.data_list = []
        self.label_list = []
        self.coord_list = []
        with open(data_list_pth, "r") as f:
            for line in f.readlines():
                l = line.strip().split(" ")
                self.data_list.append(l[0])
                self.label_list.append(int(l[1]))
                self.coord_list.append(torch.tensor([int(l[2]), int(l[3]), int(l[4]), int(l[5])]))

        # define transforms
        if self.mode == "train":
            self.transforms = T.Compose([
                T.Grayscale(3),
                T.RandomApply([T.ColorJitter(brightness=.4, contrast=.4, saturation=.2)], p=0.5),
                T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Grayscale(3),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        # extract base image
        im_path = self.data_list[item]
        im = cv2.imread(im_path)[:, :, ::-1]
        label = self.label_list[item]
        coord = self.coord_list[item]

        # generate positive sample
        im1, st1, ed1 = random_sized_crop(im, 224, area_frac=0.25)
        # im1, st1, ed1 = perspective(im1, st1, ed1)
        im1 = torch.tensor(im1).permute(2, 0, 1).float()
        vis_im1 = copy.deepcopy(im1)
        im1 = self.transforms(im1)

        im2, st2, ed2 = random_sized_crop(im, 224, area_frac=0.25)
        # im2, st2, ed2 = perspective(im2, st2, ed2)
        im2 = torch.tensor(im2).permute(2, 0, 1).float()
        vis_im2 = copy.deepcopy(im2)
        im2 = self.transforms(im2)

        M1 = cv2.getPerspectiveTransform(ed1, st1)
        M2 = cv2.getPerspectiveTransform(st2, ed2)
        M = np.matmul(M2, M1)
        M = torch.tensor(M).float()

        # fetch negative sample
        neg_im_path = random.sample(self.data_list, 1)[0]
        neg_im = cv2.imread(neg_im_path)[:, :, ::-1]
        im3, _, _ = random_sized_crop(neg_im, 224, area_frac=0.5)
        im3 = torch.tensor(im3).permute(2, 0, 1).float()
        vis_im3 = copy.deepcopy(im3)
        im3 = self.transforms(im3)

        if not self.visual:
            return im1, im2, im3, M, label
        else:
            return im1, im2, im3, M, label, im_path, coord, vis_im1, vis_im2, vis_im3


if __name__ == '__main__':
    dataset = Dataset('./dataset/dataset_v2', './dataset/dataset_v2/train.txt', visual=True)
    length = len(dataset)
    idx = np.random.randint(0, length)
    im1, im2, im3, M, label, im_path, coord, vis_im1, vis_im2, vis_im3 = dataset.__getitem__(idx)

    plt.figure(figsize=(4, 16))
    plt.subplot(4, 1, 1)
    plt.imshow(vis_im1.long().permute(1, 2, 0).numpy())

    im2_to_im1 = calculate_grid_sample(
        vis_im2.unsqueeze(0), M.unsqueeze(0), vis_im1.unsqueeze(0).size()
    )
    im2_to_im1 = im2_to_im1[0]
    plt.subplot(4, 1, 2)
    plt.imshow(im2_to_im1.long().permute(1, 2, 0).numpy())

    plt.subplot(4, 1, 3)
    plt.imshow(vis_im2.long().permute(1, 2, 0).numpy())

    plt.subplot(4, 1, 4)
    plt.imshow(vis_im3.long().permute(1, 2, 0).numpy())

    plt.savefig('./match/dataset.jpg')
