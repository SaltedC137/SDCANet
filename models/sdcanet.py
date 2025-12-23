import torch
import torch.nn as nn
from torch.nn import functional as F
from .res2net import res2net50_v1b_26w_4s

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# CA attention

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class StripDiffBlock(nn.Module):
    def __init__(self, in_channels, dilation = 2, k=3):
        super(StripDiffBlock, self).__init__()
        
        p = (dilation * (k - 1)) // 2

        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, k), padding=(0, p), dilation=(1, dilation)),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1), padding=(p, 0), dilation=(dilation, 1)),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        
        self.conv_std = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=dilation,dilation=dilation),
            nn.BatchNorm2d(in_channels),nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), 
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.coord_att = CoordAtt(in_channels,in_channels)

    def forward(self, high_feat, low_feat):

        high_up = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
        diff = torch.abs(high_up - low_feat)

        feat_h = self.conv_h(diff)
        feat_v = self.conv_v(diff)
        feat_std = self.conv_std(diff)

        feat_sum = feat_std + feat_h + feat_v

        combined = torch.cat([feat_sum, low_feat], dim=1)

        fused = self.fusion(combined)
        out = self.coord_att(fused)
        
        return out + low_feat


class ASPP(nn.Module):
    def __init__(self, in_channels, branch_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, dilation=dilation_rates[0], 
                      padding=dilation_rates[0], bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, dilation=dilation_rates[1], 
                      padding=dilation_rates[1], bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, dilation=dilation_rates[2], 
                      padding=dilation_rates[2], bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True))

        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.5) 
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.pool(x)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.project(out)




class SDCANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self,num_classes):
        super(SDCANet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self._patch_resnet()

        self.aspp = ASPP(in_channels=2048, branch_channels=256, out_channels=64, dilation_rates=[6, 12, 18])
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x_half_dem = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.sdb_half = StripDiffBlock(64, dilation=1)

        # denseness
        self.sdb_x5_x4 = StripDiffBlock(64, dilation=3)
        self.sdb_x4_x3 = StripDiffBlock(64, dilation=2)
        self.sdb_x3_x2 = StripDiffBlock(64, dilation=2)
        self.sdb_x2_x1 = StripDiffBlock(64, dilation=1)

        self.sdb_x5_x4_x3 = StripDiffBlock(64, dilation=2)
        self.sdb_x4_x3_x2 = StripDiffBlock(64, dilation=1)
        self.sdb_x3_x2_x1 = StripDiffBlock(64, dilation=1)

        self.sdb_x5_x4_x3_x2 = StripDiffBlock(64, dilation=1)
        self.sdb_x4_x3_x2_x1 = StripDiffBlock(64, dilation=1)

        self.sdb_x5_x4_x3_x2_x1 = StripDiffBlock(64, dilation=1)
        #

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.final_cls = nn.Conv2d(64,num_classes,kernel_size=1)


    def _patch_resnet(self):
        # ---  Layer 3 (target: Output Stride 8, Dilation 2) ---
        for i, block in enumerate(self.resnet.layer3):
            for conv in block.convs:
                conv.stride = (1, 1)
                conv.dilation = (2, 2)
                conv.padding = (2, 2)
            
            if i == 0:
                if hasattr(block, 'pool'):
                    block.pool.stride = 1
                    block.pool.padding = 1 
                
                if block.downsample is not None:
                    if isinstance(block.downsample[0], nn.AvgPool2d):
                        block.downsample[0].stride = 1
                        block.downsample[0].kernel_size = 1 

        # ---  Layer 4 (target: Output Stride 8, Dilation 4) ---
        for i, block in enumerate(self.resnet.layer4):
            for conv in block.convs:
                conv.stride = (1, 1)
                conv.dilation = (4, 4)
                conv.padding = (4, 4)
            if i == 0:
                if hasattr(block, 'pool'):
                    block.pool.stride = 1
                    block.pool.padding = 1
                
                if block.downsample is not None:
                    if isinstance(block.downsample[0], nn.AvgPool2d):
                        block.downsample[0].stride = 1
                        block.downsample[0].kernel_size = 1
                
    def forward(self, x):
        input_size = x.size()[2:]


        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x_half = self.resnet.relu(x)

        x1 = self.resnet.maxpool(x_half)      # bs, 64, 88, 88
        x2 = self.resnet.layer1(x1)      # bs, 256, 88, 88
        x3 = self.resnet.layer2(x2)     # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)     # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)     # bs, 2048, 11, 11

        x5_dem_1 = self.aspp(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_4 = self.sdb_x5_x4(x5_dem_1, x4_dem_1)
        x4_3 = self.sdb_x4_x3(x4_dem_1, x3_dem_1)
        x3_2 = self.sdb_x3_x2(x3_dem_1, x2_dem_1)
        x2_1 = self.sdb_x2_x1(x2_dem_1, x1)


        x5_4_3 = self.sdb_x5_x4_x3(x5_4, x4_3)
        x4_3_2 = self.sdb_x4_x3_x2(x4_3, x3_2)
        x3_2_1 = self.sdb_x3_x2_x1(x3_2, x2_1)


        x5_4_3_2 = self.sdb_x5_x4_x3_x2(x5_4_3, x4_3_2)
        x4_3_2_1 = self.sdb_x4_x3_x2_x1(x4_3_2, x3_2_1)

        x5_dem_4 = self.x5_dem_4(x5_4_3_2) 
        x5_4_3_2_1 = self.sdb_x5_x4_x3_x2_x1(x5_dem_4, x4_3_2_1)

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        out4_feat = F.interpolate(x5_dem_5, size=level4.size()[2:], mode='bilinear', align_corners=True) + level4
        out4 = self.output4(out4_feat)
        
        out3_feat = F.interpolate(out4, size=level3.size()[2:], mode='bilinear', align_corners=True) + level3
        out3 = self.output3(out3_feat)
        
        out2_feat = F.interpolate(out3, size=level2.size()[2:], mode='bilinear', align_corners=True) + level2
        out2 = self.output2(out2_feat)
        
        out1_feat = F.interpolate(out2, size=level1.size()[2:], mode='bilinear', align_corners=True) + level1
        out1 = self.output1(out1_feat) # 1/4

        out_half = self.sdb_half(out1, self.x_half_dem(x_half))
        out_final = self.final_cls(out_half)
        output = F.interpolate(out_final, size=input_size, mode='bilinear', align_corners=True)
        if self.training:
            aux_out = self.final_cls(F.interpolate(level1, size=out_half.size()[2:], mode='bilinear'))
            aux_output = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
            return output,aux_output
        return output



if __name__ == '__main__':
    model = SDCANet(num_classes=2)
    model.eval() 
    dummy_input = torch.randn(2, 3, 256, 256) 
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)
    model.train()
    outputs = model(dummy_input)
    print("Training mode output count:", len(outputs))
    print("Main output shape:", outputs[0].shape)
    print("Aux output shape:", outputs[1].shape)




