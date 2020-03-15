import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from data import cityscapes, gtav, cityscapes_al, cityscapes_al_splits, camvid, camvid_al


def get_data(data_path, tr_bs, vl_bs, n_workers=0, scale_size=0, input_size=(256, 512),
             supervised=False, num_each_iter=1, only_last_labeled=False, dataset='cityscapes', test=False,
             al_algorithm='ralis', full_res=False,
             region_size=128):
    print('Loading data...')
    candidate_set = None
    input_transform, target_transform, train_joint_transform, val_joint_transform, al_train_joint_transform = \
        get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset)

    # To train pre-trained segmentation network and upper bounds.
    if supervised:
        if 'gta' in dataset:
            train_set = gtav.GTAV('fine', 'train',
                                  data_path=data_path,
                                  joint_transform=train_joint_transform,
                                  transform=input_transform,
                                  target_transform=target_transform,
                                  camvid=True if dataset == 'gta_for_camvid' else False)
            val_set = gtav.GTAV('fine', 'val',
                                data_path=data_path,
                                joint_transform=val_joint_transform,
                                transform=input_transform,
                                target_transform=target_transform,
                                camvid=True if dataset == 'gta_for_camvid' else False)
        elif dataset == 'camvid':
            train_set = camvid.Camvid('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform)
            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'camvid_subset':
            train_set = camvid.Camvid('fine', 'train',
                                      data_path=data_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'cs_upper_bound':
            train_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                  data_path=data_path,
                                                                  joint_transform=train_joint_transform,
                                                                  transform=input_transform,
                                                                  target_transform=target_transform, supervised=True)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

        elif dataset == 'cityscapes_subset':
            train_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                  data_path=data_path,
                                                                  joint_transform=train_joint_transform,
                                                                  transform=input_transform,
                                                                  target_transform=target_transform, subset=True)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

        else:
            train_set = cityscapes.CityScapes('fine', 'train',
                                              data_path=data_path,
                                              joint_transform=train_joint_transform,
                                              transform=input_transform,
                                              target_transform=target_transform)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)
    # To train AL methods
    else:
        if dataset == 'cityscapes':
            if al_algorithm == 'ralis' and not test:
                split = 'train'
            else:
                split = 'test'
            train_set = cityscapes_al.CityScapes_al('fine', 'train',
                                                    data_path=data_path,
                                                    joint_transform=train_joint_transform,
                                                    joint_transform_al=al_train_joint_transform,
                                                    transform=input_transform,
                                                    target_transform=target_transform, num_each_iter=num_each_iter,
                                                    only_last_labeled=only_last_labeled,
                                                    split=split, region_size=region_size)
            candidate_set = cityscapes_al.CityScapes_al('fine', 'train',
                                                        data_path=data_path,
                                                        joint_transform=None,
                                                        candidates_option=True,
                                                        transform=input_transform,
                                                        target_transform=target_transform, split=split,
                                                        region_size=region_size)

            val_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                data_path=data_path,
                                                                joint_transform=val_joint_transform,
                                                                transform=input_transform,
                                                                target_transform=target_transform)

        elif dataset == 'camvid':
            train_set = camvid_al.Camvid_al('fine', 'train',
                                            data_path=data_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size)
            candidate_set = camvid_al.Camvid_al('fine', 'train',
                                                data_path=data_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train' if al_algorithm == 'ralis' and not test else 'test',
                                                region_size=region_size)

            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)

    train_loader = DataLoader(train_set,
                              batch_size=tr_bs,
                              num_workers=n_workers, shuffle=True,
                              drop_last=False)

    val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            num_workers=n_workers, shuffle=False)

    return train_loader, train_set, val_loader, candidate_set


def get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if scale_size == 0:
        print('(Data loading) Not scaling the data')
        print('(Data loading) Random crops of ' + str(input_size) + ' in training')
        print('(Data loading) No crops in validation')
        if supervised:
            train_joint_transform = joint_transforms.Compose([
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        else:
            train_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.RandomCropRegion(input_size, region_size=region_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        if (not test and al_algorithm == 'ralis') and not full_res:
            val_joint_transform = joint_transforms.Scale(1024)
        else:
            val_joint_transform = None
        al_train_joint_transform = joint_transforms.ComposeRegion([
            joint_transforms.CropRegion(region_size, region_size=region_size),
            joint_transforms.RandomHorizontallyFlip()
        ])
    else:
        print('(Data loading) Scaling training data: ' + str(
            scale_size) + ' width dimension')
        print('(Data loading) Random crops of ' + str(
            input_size) + ' in training')
        print('(Data loading) No crops nor scale_size in validation')
        if supervised:
            train_joint_transform = joint_transforms.Compose([
                joint_transforms.Scale(scale_size),
                joint_transforms.RandomCrop(input_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        else:
            train_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.Scale(scale_size),
                joint_transforms.RandomCropRegion(input_size, region_size=region_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        al_train_joint_transform = joint_transforms.ComposeRegion([
            joint_transforms.Scale(scale_size),
            joint_transforms.CropRegion(region_size, region_size=region_size),
            joint_transforms.RandomHorizontallyFlip()
        ])
        if dataset == 'gta_for_camvid':
            val_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.Scale(scale_size)])
        else:
            val_joint_transform = None
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()

    return input_transform, target_transform, train_joint_transform, val_joint_transform, al_train_joint_transform
