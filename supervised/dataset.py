import math
import pickle
import numpy as np
import cv2 as cv
from PIL import Image

import torch
import torchvision.transforms as T

from fp_utils import *


def _process(fingerprint):
    # Calculate the local gradient (using Sobel filters)
    gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)

    # Calculate the magnitude of the gradient for each pixel
    gx2, gy2 = gx ** 2, gy ** 2
    gm = np.sqrt(gx2 + gy2)

    # Integral over a square window
    sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize=False)

    # Use a simple threshold for segmenting the fingerprint pattern
    thr = sum_gm.max() * 0
    mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)

    W = (11, 11)
    gxx = cv.boxFilter(gx2, -1, W, normalize=False)
    gyy = cv.boxFilter(gy2, -1, W, normalize=False)
    gxy = cv.boxFilter(gx * gy, -1, W, normalize=False)
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy

    orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2  # '-' to adjust for y axis direction
    sum_gxx_gyy = gxx + gyy
    strengths = np.divide(cv.sqrt((gxx_gyy ** 2 + gxy2 ** 2)), sum_gxx_gyy, out=np.zeros_like(gxx),
                          where=sum_gxx_gyy != 0)

    region = fingerprint[10:90, 80:130]
    # before computing the x-signature, the region is smoothed to reduce noise
    smoothed = cv.blur(region, (5, 5), -1)
    xs = np.sum(smoothed, 1)  # the x-signature of the region

    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

    # Calculate all the distances between consecutive peaks
    distances = local_maxima[1:] - local_maxima[:-1]

    # Estimate the ridge line period as the average of the above distances
    if distances.shape[0] != 0:
        ridge_period = np.average(distances)
    else:
        ridge_period = 20

    # Create the filter bank
    or_count = 32
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]

    # Filter the whole image with each filter
    # Note that the negative image is actually used, to have white ridges on a black background as a result
    nf = 255 - fingerprint
    all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])

    y_coords, x_coords = np.indices(fingerprint.shape)
    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    # Convert to gray scale and apply the mask
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
    enhanced = cv.blur(enhanced, (7, 7))

    # Binarization
    _, ridge_lines = cv.threshold(enhanced, 64, 255, cv.THRESH_BINARY)

    # Thinning
    skeleton = cv.ximgproc.thinning(ridge_lines, thinningType=cv.ximgproc.THINNING_GUOHALL)

    # Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
    cn_filter = np.array([[1, 2, 4],
                          [128, 0, 8],
                          [64, 32, 16]
                          ])

    # Create a lookup table that maps each byte value to the corresponding crossing number
    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

    # Skeleton: from 0/255 to 0/1 values
    skeleton01 = np.where(skeleton != 0, 1, 0).astype(np.uint8)
    # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255]
    cn_values = cv.filter2D(skeleton01, -1, cn_filter, borderType=cv.BORDER_CONSTANT)
    # Apply the lookup table to obtain the crossing number of each pixel
    cn = cv.LUT(cn_values, cn_lut)
    # Keep only crossing numbers on the skeleton
    cn[skeleton == 0] = 0

    # crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
    minutiae = [(x, y, cn[y, x] == 1) for y, x in zip(*np.where(np.isin(cn, [1, 3])))]

    # A 1-pixel background border is added to the mask before computing the distance transform
    mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[
                    1:-1, 1:-1]

    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]] > 3, minutiae))

    return ridge_lines, skeleton, filtered_minutiae


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list_pth, mode="train", visual=False):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.data_list_pth = data_list_pth
        self.visual = visual
        self.mode = mode

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
                T.Resize((224, 224)),
                T.Grayscale(3),
                T.RandomAffine(degrees=(0, 0), translate=(0.25, 0.25)),
                T.RandomApply([T.ColorJitter(brightness=.4, contrast=.4, saturation=.2)], p=0.5),
                T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        im_pth = self.data_list[item]
        im = Image.open(im_pth)
        im = self.transforms(im)
        label = self.label_list[item]
        coord = self.coord_list[item]
        if self.visual:
            return im, label, im_pth, coord
        else:
            return im, label


class Dataset1K(torch.utils.data.Dataset):
    def __init__(self, train_test_split, mode="train", return_type=False):
        super(Dataset1K, self).__init__()
        self.train_test_split = train_test_split
        self.mode = mode
        self.return_type = return_type

        # read data_list
        with open(self.train_test_split, "rb") as f:
            data_dict = pickle.load(f)[self.mode]

        self.data_list = data_dict["standard"]["img_pth"] + data_dict["blur"]["img_pth"] + \
                    data_dict["water"]["img_pth"] + data_dict["heat"]["img_pth"]
        self.label_list = data_dict["standard"]["label"] + data_dict["blur"]["label"] + \
                    data_dict["water"]["label"] + data_dict["heat"]["label"]

        # define transforms
        if self.mode == "train":
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(3),
                T.RandomAffine(degrees=(0, 0), translate=(0.25, 0.25)),
                T.RandomApply([T.ColorJitter(brightness=.4, contrast=.4, saturation=.2)], p=0.5),
                T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        im_pth = self.data_list[item]
        im = Image.open(im_pth)
        im_type = im_pth.split("/")[-2]
        im = self.transforms(im)
        label = self.label_list[item]
        if self.return_type:
            return im, label, im_type
        else:
            return im, label


if __name__ == "__main__":
    dataset = Dataset("dataset/dataset_v2", "dataset/dataset_v2/train.txt")
