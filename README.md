# Reinforced Active Learning for Image Segmentation (RALIS)
Code for the paper [Reinforced Active Learning for Image Segmentation](https://arxiv.org/abs/2002.06583)

## Dependencies 
- python 3.6.5
- numpy 1.14.5
- scipy 1.1.0
- Pytorch 0.4.0

## Scripts
The folder 'scripts' contains the different bash scripts that could be used to train the same models used in the paper, for both Camvid and Cityscapes datasets. 
- launch_supervised.sh: To train the pretrained segmentation models. 
- launch_baseline.sh: To train the baselines 'random', 'entropy' and 'bald'.
- launch_train_ralis.sh: To train the 'ralis' model.
- launch_test_ralis.sh: To test the 'ralis' model. 

## Datasets
Camvid: https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

Cityscapes: https://www.cityscapes-dataset.com/

## Trained models
To download the trained RALIS models for Camvid and Cityscapes (as well as the pretrained segmentation model on GTA and D_T subsets): https://drive.google.com/file/d/13C4e0bWw6SEjTAD7JdAfLGVz7p7Veeb9/view?usp=sharing
## Citation
If you use this code, please cite:
```
@inproceedings{
Casanova2020Reinforced,
title={Reinforced active learning for image segmentation},
author={Arantxa Casanova and Pedro O. Pinheiro and Negar Rostamzadeh and Christopher J. Pal},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SkgC6TNFvr}
}
```
