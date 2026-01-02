import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import warnings
warnings.filterwarnings("ignore")

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        
        if activation == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'silu':
            act_layer = nn.SiLU(inplace=True)
        else:
            act_layer = nn.ReLU(inplace=True)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_layer,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_layer
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class EfficientUNet(nn.Module):
    def __init__(self, encoder_name='efficientnet-b4', num_classes=1, use_mixed_activation=False):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = EfficientNet.from_pretrained(encoder_name)
        
        dummy = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            feats = self.encoder.extract_endpoints(dummy)
            ch1 = feats['reduction_1'].shape[1]  # ~H/2
            ch2 = feats['reduction_2'].shape[1]  # ~H/4
            ch3 = feats['reduction_3'].shape[1]  # ~H/8
            ch4 = feats['reduction_4'].shape[1]  # ~H/16
            ch5 = feats['reduction_5'].shape[1]  # ~H/32
        # decoder
        if use_mixed_activation:

            self.up4 = UpBlock(ch5 + ch4, ch4, activation='relu')       
            self.up3 = UpBlock(ch4 + ch3, ch3, activation='relu')       
            self.up2 = UpBlock(ch3 + ch2, ch2, activation='leaky_relu') 
            self.up1 = UpBlock(ch2 + ch1, ch1, activation='silu')       
        else:
            # ReLU
            self.up4 = UpBlock(ch5 + ch4, ch4, activation='relu')
            self.up3 = UpBlock(ch4 + ch3, ch3, activation='relu')
            self.up2 = UpBlock(ch3 + ch2, ch2, activation='relu')
            self.up1 = UpBlock(ch2 + ch1, ch1, activation='relu')
        

        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(ch1, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        if H % 32 != 0 or W % 32 != 0:
            # 32 
            target_H = ((H + 31) // 32) * 32
            target_W = ((W + 31) // 32) * 32
            x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=True)

        # multi scale
        feats = self.encoder.extract_endpoints(x)
        e1 = feats['reduction_1']  # H/2
        e2 = feats['reduction_2']  # H/4
        e3 = feats['reduction_3']  # H/8
        e4 = feats['reduction_4']  # H/16
        e5 = feats['reduction_5']  # H/32

        # decoder
        d4 = self.up4(e5, e4)      # H/16
        d3 = self.up3(d4, e3)      # H/8
        d2 = self.up2(d3, e2)      # H/4
        d1 = self.up1(d2, e1)      # H/2
        
        # output
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        

        if out.shape[2] != H or out.shape[3] != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
            
        return out
    
if __name__ == "__main__":
    model = EfficientUNet(num_classes=2)
    model.eval()
    with torch.no_grad():
        input = torch.randn(2, 3, 256, 256)
    out = model(input)
    print(out.shape)