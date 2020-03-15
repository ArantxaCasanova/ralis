import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from copy import deepcopy

num_classes = 19
ignore_label = 19
path = 'datasets/cityscapes'
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153,
           153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset(quality, mode, root):
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' \
            else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = ''
        mask_path = os.path.join(root, '', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)

    categories = os.listdir(img_path)
    categories.sort()
    categories_m = os.listdir(mask_path)
    categories_m.sort()

    assert categories == categories_m
    items = []

    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
                    os.path.join(mask_path, c, it + mask_postfix), it)
            items.append(item)
    return items


class CityScapes_al(data.Dataset):
    def __init__(self, quality, mode, data_path='', joint_transform=None, joint_transform_al=None,
                 sliding_crop=None, transform=None, target_transform=None, candidates_option=False,
                 region_size=(128, 128),
                 num_each_iter=1, only_last_labeled=True, split='train'):
        self.num_each_iter = num_each_iter
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.root = data_path + path
        self.imgs = make_dataset(quality, mode, self.root)

        splits = np.load('data/cityscapes_al_splits.npy', allow_pickle=True).item()
        self.state_subset = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_s'])]
        self.state_subset_regions = {}
        for i in range(len(splits['d_s'])):
            x_r1 = np.arange(0, 2048 - region_size[0] + 1, region_size[0])
            y_r1 = np.arange(0, 1024 - region_size[1] + 1, region_size[1])
            self.state_subset_regions.update({i: np.array(np.meshgrid(x_r1, y_r1)).T.reshape(-1, 2)})

        if split == 'train':
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_t'])]
        elif split == 'test':
            self.imgs = [img for i, img in enumerate(self.imgs) if (img[-1] in splits['d_v'])]

        print('Using ' + str(split) + ' splitting of ' + str(len(self.imgs)) + ' images.')

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.joint_transform_al = joint_transform_al
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label,
                              11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9,
                              23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

        self.balance_cl = []
        self.only_last_labeled = only_last_labeled
        self.candidates = candidates_option
        self.end_al = False
        self.selected_images = []
        self.selected_regions = dict()
        self.list_regions = []
        self.num_imgs = len(self.imgs)
        splitters_x = np.arange(0, 2048 - region_size[0] + 1, region_size[0])
        splitters_y = np.arange(0, 1024 - region_size[1] + 1, region_size[1])
        splitters_mesh = np.array(np.meshgrid(splitters_y, splitters_x)).T.reshape(-1, 2)
        prov_splitters = splitters_mesh.copy()
        prov_splitters_x = list(prov_splitters[:, 1])
        prov_splitters_y = list(prov_splitters[:, 0])
        self.unlabeled_regions_x = [deepcopy(prov_splitters_x) for _ in range(self.num_imgs)]
        self.unlabeled_regions_y = [deepcopy(prov_splitters_y) for _ in range(self.num_imgs)]
        self.num_unlabeled_regions_total = (1024 // region_size[1]) * (2048 // region_size[0]) * self.num_imgs
        self.region_size = region_size

    def get_subset_state(self, index):
        img_path, mask_path, im_name = self.state_subset[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        mask = np.array(mask)
        mask_copy = mask.copy()

        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, None, (img_path, mask_path, im_name), self.state_subset_regions[index]

    def __getitem__(self, index):
        # Train with all labeled images, selecting a random region per image, and doing the random crop around it
        if self.candidates or self.end_al:
            img_path, mask_path, im_name = self.imgs[self.selected_images[index]]
            # Select random region in the image to make sure there is a region in the crop
            selected_region_ind = np.random.choice(len(self.selected_regions[self.selected_images[index]]))
            selected_region = self.selected_regions[self.selected_images[index]][selected_region_ind]
            selected = [self.selected_images[index]]
        else:
            # Train with just the last regions selected, random crop around the selected region
            if self.only_last_labeled:
                selected = self.list_regions[len(self.list_regions) - self.num_each_iter:][index]
            # Train with all labeled regions so far, random crop around the selected region
            else:
                selected = self.list_regions[index]
            img_path, mask_path, im_name = self.imgs[selected[0]]
            # img_path, mask_path, im_name = self.imgs[self.selected_images[index]]

            selected_region = selected[1]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v

        if not self.candidates:
            mask_copy = self.maskout_unselected_regions(mask_copy, selected[0], self.region_size)

        mask = Image.fromarray(mask_copy.astype(np.uint8))
        if self.joint_transform is not None:
            if not self.candidates:
                img, mask = self.joint_transform(img, mask, selected_region)
            else:
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
            return img, mask, (img_path, mask_path, im_name), selected_region[0] if not self.candidates else \
            self.selected_images[index], 0

    def maskout_unselected_regions(self, mask, image, region_size=(128, 128)):
        masked = np.full(mask.shape, ignore_label)
        for region in self.selected_regions[image]:
            # Indexes reverted, because here width is the 2nd index.
            r_x = int(region[1])
            r_y = int(region[0])
            masked[r_x: r_x + region_size[1], r_y: r_y + region_size[0]] = mask[r_x: r_x + region_size[1],
                                                                           r_y: r_y + region_size[0]]
        return masked

    def get_specific_item(self, path):
        img_path, mask_path, im_name = self.imgs[path]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
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
            return img, mask, None, (img_path, mask_path, im_name)

    def __len__(self):
        if self.candidates or self.end_al:
            return len(self.selected_images)
        else:
            if self.only_last_labeled:
                return self.num_each_iter
            else:
                return len(self.list_regions)

    def get_random_unlabeled_region_image(self, index):
        counter_i = int(np.random.choice(range(len(self.unlabeled_regions_x[index])), 1, replace=False))
        counter_x = self.unlabeled_regions_x[index].pop(counter_i)
        counter_y = self.unlabeled_regions_y[index].pop(counter_i)
        return counter_x, counter_y

    def labeled_set(self):
        return self.selected_images

    def get_labeled_regions(self):
        return self.selected_regions

    def get_unlabeled_regions(self):
        return deepcopy(self.unlabeled_regions_x), deepcopy(self.unlabeled_regions_y)

    def set_unlabeled_regions(self, rx, ry):
        self.unlabeled_regions_x = rx
        self.unlabeled_regions_y = ry

    def get_num_unlabeled_regions(self, region_size=128):
        return self.num_unlabeled_regions_total

    def get_num_unlabeled_regions_image(self, index):
        return len(self.unlabeled_regions_x[index])

    def get_num_labeled_regions(self):
        labeled_regions = 0
        for key, value in self.selected_regions.items():
            labeled_regions += len(value)
        return labeled_regions

    def get_candidates(self, num_regions_unlab=1000):
        """Take all regions from an image into account. We will take as many images as we need to get to the number
        of regions we want to have as candidates.
        :param num_regions_unlab: (int) number of unlabeled regions to form candidates.
        :return: list of images that have, in total, the target number of unlabeled regions (num_regions_unlab)
        """
        unlabeled_regions = 0
        candidates = []
        images_list = list(range(self.num_imgs))
        while unlabeled_regions <= num_regions_unlab:
            if len(images_list) == 0:
                raise ValueError('There is no more unlabeled regions to fullfill the amount we want!')
            index = np.random.choice(len(images_list))
            candidate = images_list.pop(index)
            num_regions_left = self.get_num_unlabeled_regions_image(int(candidate))
            if num_regions_left > 0:
                unlabeled_regions += num_regions_left
                candidates.append(candidate)
        return candidates

    def check_class_region(self, img, region, region_size=(128, 128), eps=1E-7):
        img_path, mask_path, im_name = self.imgs[img]
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = mask_copy.astype(np.uint8)
        r_x = int(region[1])
        r_y = int(region[0])
        region_classes = mask[r_x: r_x + region_size[1], r_y: r_y + region_size[0]]
        unique, counts = np.unique(region_classes, return_counts=True)
        balance = []
        for cl in range(0, 20):
            if cl in unique:
                balance.append(counts[unique == cl].item() / counts.sum())
            else:
                balance.append(eps)
        self.balance_cl.append(balance)

    def add_index(self, paths, region=None):
        if isinstance(paths, list):
            for path in paths:
                if path not in self.selected_images:
                    self.selected_images.append(int(path))
                if region is not None:
                    if int(path) in self.selected_regions.keys():
                        if region not in self.selected_regions[int(path)]:
                            self.selected_regions[int(path)].append(region)
                            self.add_index_(path, region)
                    else:
                        self.selected_regions.update({int(path): [region]})
                        self.add_index_(path, region)

        else:
            if paths not in self.selected_images:
                self.selected_images.append(int(paths))
            if region is not None:
                if int(paths) in self.selected_regions.keys():
                    if region not in self.selected_regions[int(paths)]:
                        self.selected_regions[int(paths)].append(region)
                        self.add_index_(paths, region)

                    else:
                        print('Region already added!')
                else:
                    self.selected_regions.update({int(paths): [region]})
                    self.add_index_(paths, region)

    def add_index_(self, path, region):
        self.list_regions.append((int(path), region))
        self.num_unlabeled_regions_total -= 1

        self.check_class_region(int(path), (region[0], region[1]), self.region_size)
        for i in range(len(self.unlabeled_regions_x[int(path)])):
            if self.unlabeled_regions_x[int(path)][i] == region[0] and \
                    self.unlabeled_regions_y[int(path)][i] == region[1]:
                self.unlabeled_regions_x[int(path)].pop(i)
                self.unlabeled_regions_y[int(path)].pop(i)
                break

    def del_index(self, paths):
        self.selected_images.remove(paths)

    def reset(self):
        self.selected_images = []
