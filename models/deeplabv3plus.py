import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from .backbones.mobilenet import mobilenet_v2


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[0],
                               padding=dilation_rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[1],
                               padding=dilation_rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[2],
                               padding=dilation_rates[2])
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = self.conv5(x5)
        x5 = torch.nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        return torch.cat((x1, x2, x3, x4, x5), dim=1)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        # self.resnet = models.resnet50(pretrained=False) 
        self.resnet = mobilenet_v2(pretrained=False)

        self.layer0 = nn.Sequential(self.resnet.features[0], self.resnet.features[1])
        self.layer1 = nn.Sequential(self.resnet.features[2], self.resnet.features[3], self.resnet.features[4],
                                    self.resnet.features[5], self.resnet.features[6])
        self.layer2 = nn.Sequential(self.resnet.features[7], self.resnet.features[8], self.resnet.features[9],
                                    self.resnet.features[10])
        self.layer3 = nn.Sequential(self.resnet.features[11], self.resnet.features[12], self.resnet.features[13],
                                    self.resnet.features[14], self.resnet.features[15], self.resnet.features[16],
                                    self.resnet.features[17])
        self.layer4 = nn.Sequential(self.resnet.features[18])
        
        self.aspp = ASPP(in_channels=1280, out_channels=256, dilation_rates=[6, 12, 18])
        # fix
        self.ca1 = ChannelAttention(in_planes=1280)  # fix to match aspp

        self.sa1 = SpatialAttention()
        self.decoder = nn.Sequential(
            nn.Conv2d(256 * 5, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_aspp = self.aspp(x4)
        x_aspp = self.ca1(x_aspp) * x_aspp
        x_aspp = self.sa1(x_aspp) * x_aspp
        x_decoder = self.decoder(x_aspp)
        x_decoder = torch.nn.functional.interpolate(x_decoder, size=x.size()[2:], mode='bilinear', align_corners=False)
        return x_decoder
    

if __name__ == '__main__':
    # model = DeepLabV3Plus(2)
    # input = torch.randn(size=(1,3,256,256))
    # out = model(input)
    # print(out.shape)

    # model = DeepLabV3Plus2(2)
    # input = torch.randn(size=(1, 3, 256, 256))
    # out = model(input)
    # print(out.shape)

    model = DeepLabV3Plus(2)
    input = torch.randn(size=(1, 3, 256, 256))
    out = model(input)
    print(out.shape)