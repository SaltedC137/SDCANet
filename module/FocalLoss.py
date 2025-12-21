import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class':
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if alpha is not None:
                if isinstance(alpha, list):
                    self.alpha = torch.tensor(alpha)
                elif isinstance(alpha, (float, int)):
                    self.alpha = torch.tensor([alpha] * num_classes)
                else:
                    self.alpha = alpha
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        # inputs: [B, 1, H, W], targets: [B, 1, H, W]
        probs = torch.sigmoid(inputs)
        
        if targets.dtype != torch.float32:
            targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:

            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ 
        Focal loss for multi-class semantic segmentation. 
        inputs: [B, C, H, W] (Logits)
        targets: [B, H, W] (Long/Int, values in 0 ~ C-1)
        """
        
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        log_probs = torch.log(probs + 1e-8)
        focal_weight = (1 - probs) ** self.gamma

        #  Loss
        #  - alpha * (1-pt)^gamma * log(pt) * target_one_hot
        loss = -targets_one_hot * focal_weight * log_probs


        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_view = self.alpha.view(1, -1, 1, 1)
            loss = loss * alpha_view

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:

            if isinstance(self.alpha, (list, torch.Tensor)):
                 alpha = torch.tensor(self.alpha).to(inputs.device)
                 if inputs.ndim == 4:
                     alpha = alpha.view(1, -1, 1, 1)
                 else:
                     alpha = alpha.view(1, -1)
                 
                 alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                 bce_loss = alpha_t * bce_loss
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss