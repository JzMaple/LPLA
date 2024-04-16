from PIL import Image

import torch
import torchvision.transforms as T


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list_pth, n_views=2, mode="train", visual=False):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.data_list_pth = data_list_pth
        self.n_views = n_views
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
                T.Resize((224, 224)),
                T.Grayscale(3),
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
        im = [self.transforms(im) for _ in range(self.n_views)]
        label = self.label_list[item]
        coord = self.coord_list[item]
        if self.visual:
            return im, label, im_pth, coord
        else:
            return im, label
