import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

path = '/datasets/gta5'
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153,
           153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset(mode, root):
    mask_path = os.path.join(root, 'labels')
    img_path = os.path.join(root, 'images')
    items = []
    # Import predefined splits
    import scipy.io as cpio
    split = cpio.loadmat(os.path.join(root, 'split.mat'))

    if mode in ['train', 'val', 'test']:
        splits = split[mode + "Ids"][:, 0].tolist()
    elif mode in ['trainval']:
        splits = split["trainIds"][:, 0].tolist() + split["valIds"][:,
                                                    0].tolist()
    elif mode in ['all']:
        splits = split["trainIds"][:, 0].tolist() + split["valIds"][:,
                                                    0].tolist() + split[
                                                                      "testIds"][
                                                                  :,
                                                                  0].tolist()
    else:
        raise ValueError('Split selected does not exist')

    # Indexes with mismatching sizes between image and mask
    to_ignore = [15188, 17705] + [*range(20801, 20861)]

    for it in splits:
        if it not in to_ignore:
            item = (os.path.join(img_path, '%05d.png' % it),
                    os.path.join(mask_path, '%05d.png' % it), str(it))
            items.append(item)
    print('GTAV ' + mode + ' split has %d images' % len(items))
    return items


class GTAV(data.Dataset):
    def __init__(self, quality, mode, data_path='', joint_transform=None,
                 sliding_crop=None, transform=None, target_transform=None, camvid=False):

        self.root = data_path + path
        self.imgs = make_dataset(mode, self.root)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        if camvid:
            self.num_classes = 11
            self.ignore_label = 11
            self.id_to_trainid = {0: self.ignore_label, 1: self.ignore_label,
                                  2: self.ignore_label,
                                  3: self.ignore_label, 4: self.ignore_label,
                                  5: self.ignore_label, 6: self.ignore_label,
                                  7: 3, 8: 4, 9: self.ignore_label, 10: self.ignore_label,
                                  11: 1, 12: self.ignore_label, 13: 7,
                                  14: self.ignore_label, 15: self.ignore_label,
                                  16: self.ignore_label, 17: 2,
                                  18: self.ignore_label, 19: 6, 20: 6, 21: 5, 22: self.ignore_label,
                                  23: 0, 24: 9, 25: self.ignore_label, 26: 8, 27: 8,
                                  28: 8, 29: self.ignore_label, 30: self.ignore_label,
                                  31: self.ignore_label, 32: 8, 33: 10, 34: self.ignore_label}
        else:
            self.num_classes = 19
            self.ignore_label = 19
            self.id_to_trainid = {0: self.ignore_label, 1: self.ignore_label,
                                  2: self.ignore_label,
                                  3: self.ignore_label, 4: self.ignore_label,
                                  5: self.ignore_label, 6: self.ignore_label,
                                  7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label,
                                  11: 2, 12: 3, 13: 4,
                                  14: self.ignore_label, 15: self.ignore_label,
                                  16: self.ignore_label, 17: 5,
                                  18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9,
                                  23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                                  28: 15, 29: self.ignore_label, 30: self.ignore_label,
                                  31: 16, 32: 17, 33: 18, 34: self.ignore_label}

    def __getitem__(self, index):
        img_path, mask_path, im_name = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        if img.size != mask.size:
            print(im_name)
            print(img.size)
            print(mask.size)
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info), im_name
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask, im_name

    def __len__(self):
        return len(self.imgs)
