import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_focal=0.5, 
                 focal_alpha=0.25, focal_gamma=2.0, smooth=1e-6,
                 class_num=1):
        """
        Args:
            class_num: 1 for binary, >1 for multi-class
        """
        super(DiceFocalLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.smooth = smooth
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.class_num = class_num

    def dice_loss(self, inputs, targets):
        """Dice Loss"""
        if self.class_num == 1:
            inputs = torch.sigmoid(inputs).squeeze(1)
            targets = targets.float()
        else:
            inputs = F.softmax(inputs, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=self.class_num).permute(0, 3, 1, 2).float()
            
            dice = 0
            for i in range(self.class_num):
                input_i = inputs[:, i]
                target_i = targets_one_hot[:, i]
                intersection = (input_i * target_i).sum()
                union = input_i.sum() + target_i.sum()
                dice_i = (2. * intersection + self.smooth) / (union + self.smooth)
                dice += (1 - dice_i)
            return dice / self.class_num
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

    def focal_loss(self, inputs, targets):
        """Focal Loss"""
        if self.class_num == 1:
            inputs = torch.sigmoid(inputs).squeeze(1)
            targets = targets.float()
            
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * BCE_loss
            
            return focal_loss.mean()
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss

            return focal_loss.mean()

    def forward(self, inputs, targets):
        dice_loss_val = self.dice_loss(inputs, targets)
        focal_loss_val = self.focal_loss(inputs, targets)
        
        total_loss = (self.weight_dice * dice_loss_val + 
                     self.weight_focal * focal_loss_val)
        
        return total_loss