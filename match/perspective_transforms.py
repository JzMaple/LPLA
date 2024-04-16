import cv2
import math
import torch
import copy
import numpy as np
import torch.nn.functional as F


def scale(size, im):
    """Performs scaling (HWC format)."""
    h, w = im.shape[:2]
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    h_new, w_new = size, size
    if w < h:
        h_new = int(math.floor((float(h) / w) * size))
    else:
        w_new = int(math.floor((float(w) / h) * size))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.uint8)


def center_crop(size, _im):
    """Performs center cropping (HWC format)."""
    im = scale(size, _im)
    h, w = im.shape[:2]
    y = int(math.ceil((h - size) / 2))
    x = int(math.ceil((w - size) / 2))
    im_crop = im[y: (y + size), x: (x + size), :]
    assert im_crop.shape[:2] == (size, size)
    st_points = np.array([[y, x], [y, x + size], [y + size, x], [y + size, x + size]], dtype=np.float32)
    st_points = st_points * _im.shape[0] / im.shape[0]
    ed_points = np.array([[0, 0], [0, size], [size, 0], [size, size]], dtype=np.float32)
    return im_crop.astype(np.uint8), st_points, ed_points


def random_sized_crop(im, size, area_frac=0.08, max_iter=10):
    """Performs Inception-style cropping (HWC format)."""
    h, w = im.shape[:2]

    area = h * w
    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h_crop = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w_crop, h_crop = h_crop, w_crop
        if h_crop <= h and w_crop <= w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            im_crop = im[y: (y + h_crop), x: (x + w_crop), :]
            assert im_crop.shape[:2] == (h_crop, w_crop)
            im_crop = cv2.resize(im_crop, (size, size), interpolation=cv2.INTER_LINEAR)
            st_points = np.array([[y, x], [y, x + w_crop], [y + h_crop, x], [y + h_crop, x + w_crop]], dtype=np.float32)
            ed_points = np.array([[0, 0], [0, size], [size, 0], [size, size]], dtype=np.float32)
            return im_crop.astype(np.uint8), st_points, ed_points
    return center_crop(size, im)


def perspective(im, st_points, ed_points):
    H, W, _ = im.shape
    st = ed_points
    ed = ed_points + np.random.randint(-50, 50, size=(4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(st[:, ::-1], ed[:, ::-1])
    im_contrast = cv2.warpPerspective(im, M, (W, H))
    ed_points = ed
    return im_contrast, st_points, ed_points


def calculate_grid_sample(feat, M, size, mode='bilinear', default_size=None):
    device = feat.device
    B, C, H, W = feat.size()
    B, _C, _H, _W = size
    if default_size is None:
        dh, dw = 224, 224
    else:
        dh, dw = default_size
    x_grid = (torch.arange(_W).reshape(1, _W, 1).repeat(_H, 1, 1).float() + 0.5) / _W * dw
    y_grid = (torch.arange(_H).reshape(_H, 1, 1).repeat(1, _W, 1).float() + 0.5) / _H * dh
    z_grid = torch.ones(_H, _W, 1).float()

    grid = torch.cat([y_grid, x_grid, z_grid], dim=2).to(device)
    grid = grid.reshape(1, _H, _W, 3).repeat(B, 1, 1, 1).reshape(B, -1, 3).permute(0, 2, 1)
    grid = torch.bmm(M, grid).permute(0, 2, 1)
    grid = grid[:, :, :2] / grid[:, :, 2:3]
    # save grid for visualization
    # return_grid = copy.deepcopy(grid.reshape(B, _H, _W, 2))
    grid[:, :, 0] = grid[:, :, 0] * 2 / dh - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / dw - 1
    grid = torch.flip(grid, dims=[2])
    grid = grid.reshape(B, _H, _W, 2)
    output = F.grid_sample(feat, grid, mode=mode, align_corners=False)

    return output
