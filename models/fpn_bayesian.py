# Adapted from https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 freezed=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if freezed:
            for i in self.bn1.parameters():
                i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if freezed:
            for i in self.bn2.parameters():
                i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if freezed:
            for i in self.bn3.parameters():
                i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.ModuleList([nn.Conv2d(inplanes,
                                                  planes * self.expansion,
                                                  kernel_size=1, stride=stride,
                                                  bias=False),
                                        nn.BatchNorm2d(
                                            planes * self.expansion)])
            if freezed:
                for i in downsample[1].parameters():
                    i.requires_grad = False
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample[0](residual)
            residual = self.downsample[1](residual)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000,
                 freezed=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if freezed:
            for i in self.bn1.parameters():
                i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], freezed=freezed)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       freezed=freezed)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       freezed=freezed)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       freezed=freezed)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, freezed=False):
        downsample = None

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, freezed=freezed))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, l in enumerate(self.layer1):
            x = l(x)
        for i, l in enumerate(self.layer2):
            x = l(x)
        for i, l in enumerate(self.layer3):
            x = l(x)
        for i, l in enumerate(self.layer4):
            x = l(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor, num_channels=128):
        super(Upsample, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.up_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                                 stride=1, padding=1)

    def crop_layer(self, x, target_size):
        dif = [(x.size()[2] - target_size[0]) // 2,
               (x.size()[3] - target_size[1]) // 2]
        cs = target_size
        return x[:, :, dif[0]:dif[0] + cs[0], dif[1]:dif[1] + cs[1]]

    def forward(self, x, target_size):
        out = self.up(x)
        out = self.crop_layer(out, target_size[2:])
        out = self.up_conv(out)
        return out


class FPN_bayesian(nn.Module):
    def __init__(self, num_classes, pretrained=True, freezed=False, which_resnet='resnet50'):
        super(FPN_bayesian, self).__init__()
        '''FPN architecture.
               Args:
                 num_classes: (int) Number of classes present in the dataset.
                 pretrained: (bool) If True, ImageNet pretraining for ResNet is
                                    used.
                 freezed: (bool) If True, batch norm is freezed.
                 which_resnet: (str) Indicates if we use ResNet50 or ResNet101.
        '''
        self.in_planes = 64
        if which_resnet == 'resnet50':
            string_load = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            resnet = ResNet(freezed=freezed)
        elif which_resnet == 'resnet101':
            string_load = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
            resnet = ResNet(freezed=freezed, layers=[3, 4, 23, 3])
        else:
            raise ValueError('ResNet type not recognized')
        if pretrained:
            pretrained_dict = model_zoo.load_url(string_load)
            state = resnet.state_dict()
            state.update(pretrained_dict)
            resnet.load_state_dict(state)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        if freezed:
            for i in self.bn1.parameters():
                i.requires_grad = False

        # Bottom-up layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
                                  padding=0)  # Reduce channels

        # Smooth layers
        self.smooth0 = self.lateral_smooth()
        self.smooth1 = self.lateral_smooth()
        self.smooth2 = self.lateral_smooth()
        self.smooth3 = self.lateral_smooth()

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
                                   padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                                   padding=0)

        # Lateral upsamples
        self.latup0 = Upsample(scale_factor=8)
        self.latup1 = Upsample(scale_factor=4)
        self.latup2 = Upsample(scale_factor=2)

        # Linear classifier
        self.classifier = nn.Conv2d(128 * 4, num_classes, kernel_size=3,
                                    stride=1, padding=1)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout()

    def lateral_smooth(self):
        layers = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, bayesian=False, n_iter=1):
        # Bottom-up
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = F.relu(c1)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = c1
        for i, l in enumerate(self.layer1):
            c2 = l(c2)
        c3 = c2
        for i, l in enumerate(self.layer2):
            c3 = l(c3)
        c4 = c3
        for i, l in enumerate(self.layer3):
            c4 = l(c4)
        c5 = c4
        for i, l in enumerate(self.layer4):
            c5 = l(c5)
        # Top-down
        p5p = self.toplayer(c5)
        p4p = self._upsample_add(p5p, self.latlayer1(c4))
        p3p = self._upsample_add(p4p, self.latlayer2(c3))
        p2 = self._upsample_add(p3p, self.latlayer3(c2))
        # Lateral smooth
        p5_ = self.smooth0(p5p)
        p4_ = self.smooth1(p4p)
        p3_ = self.smooth2(p3p)
        p2_ = self.smooth3(p2)
        # Lateral upsampling
        p5 = self.latup0(p5_, p2_.size())
        p4 = self.latup1(p4_, p2_.size())
        p3 = self.latup2(p3_, p2_.size())

        out_ = [p5, p4, p3, p2_]
        out_ = torch.cat(out_, 1)
        if bayesian:
            out = []
            for n in range(n_iter):
                elem = self.final_up(self.classifier(self.dropout(out_))).cpu().data
                out.append(elem)
                del (elem)
        else:
            out_ds = self.classifier(out_)
            out = self.final_up(out_ds)

        return out, c5


def FPN50_bayesian(num_classes, pretrained=False,
                   freezed=False):
    model = FPN_bayesian(num_classes=num_classes,
                         pretrained=pretrained, freezed=freezed)
    return model


def FPN101_bayesian(num_classes, pretrained=True,
                    freezed=False):
    model = FPN_bayesian(num_classes=num_classes,
                         pretrained=pretrained, freezed=freezed,
                         which_resnet='resnet101')
    return model
