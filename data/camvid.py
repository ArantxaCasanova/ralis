import glob
import os

import numpy as np
from PIL import Image
from torch.utils import data

num_classes = 11
ignore_label = 11
path = 'datasets/camvid'
palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 0, 0, 192, 128, 128, 0, 192, 128, 128, 64, 64, 128,
           64, 0, 128, 64, 64, 0, 0, 128, 192, 0, 0, 0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset(mode, root):
    if mode == "train":
        img_path = os.path.join(root, "train")
        mask_path = os.path.join(root, "trainannot")
    elif mode == "val":
        img_path = os.path.join(root, "val")
        mask_path = os.path.join(root, "valannot")
    elif mode == "test":
        img_path = os.path.join(root, "test")
        mask_path = os.path.join(root, "testannot")
    else:
        raise ValueError('Dataset split specified does not exist!')
    print(img_path)

    img_paths = [f for f in glob.glob(os.path.join(img_path, "*.png"))]
    items = []
    for im_p in img_paths:
        item = (im_p, os.path.join(mask_path, im_p.split('/')[-1]), im_p.split('/')[-1])
        items.append(item)
    return items


class Camvid(data.Dataset):
    def __init__(self, quality, mode, data_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, subset=False):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        d_t = np.load('data/camvid_al_splits.npy', allow_pickle=True).item()['d_t']

        if subset:
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in d_t)]
        print('Using ' + str(len(self.imgs)) + ' images.')

    def __getitem__(self, index):
        img_path, mask_path, im_name = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, (img_path, mask_path, im_name)

    def __len__(self):
        return len(self.imgs)
