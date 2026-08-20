import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model, target_layer=None):
        if isinstance(model, nn.DataParallel):
            model = model.module
        self.model = model

        if target_layer is None:
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not convs:
                raise ValueError('model has no conv layer and no target_layer given')
            target_layer = convs[-1]
        elif isinstance(target_layer, str):
            candidates = {name: m for name, m in model.named_modules()}
            if target_layer not in candidates:
                raise KeyError(f'target layer "{target_layer}" not found in model')
            target_layer = candidates[target_layer]
        elif isinstance(target_layer, int):
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not (0 <= target_layer < len(convs)):
                raise IndexError(f'target layer index {target_layer} out of range')
            target_layer = convs[target_layer]

        self.target_layer = target_layer
        self._forward_handle = None
        self._backward_handle = None
        self._activations = {}

    def _forward_hook(self, module, feat_in, feat_out):
        self._activations['x'] = feat_in[0]

    def _backward_hook(self, module, grad_in, grad_out):
        self._activations['grad'] = grad_in[0]

    def __call__(self, image_tensor, target_class=1):
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        training = self.model.training
        self.model.eval()

        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
        try:
            out = self.model(image_tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
            score = out[:, target_class].sum()
            self.model.zero_grad()
            score.backward()

            weights = self._activations['grad'].mean(dim=(2, 3)).reshape(1, -1, 1, 1)
            cam = torch.relu((weights * self._activations['x']).sum(dim=1, keepdim=True))
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.detach().cpu().squeeze().numpy()
        finally:
            self._forward_handle.remove()
            self._backward_handle.remove()
            self._forward_handle = None
            self._backward_handle = None
            self.model.train(training)


def _denormalize_image(img_tensor, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    rgb = img[..., :3] * np.array(std[:3]) + np.array(mean[:3])
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb


def save_gradcam_heatmaps(model, image_tensors, label_masks, model_name,
                          save_dir=None, mean=None, std=None, target_class=1,
                          target_layer=None):
    if save_dir is None:
        save_dir = os.path.join('./output', model_name, 'gradcam')
    os.makedirs(save_dir, exist_ok=True)

    gradcam = GradCAM(model, target_layer)
    kernel = np.ones((3, 3), np.uint8)

    for idx, (img_tensor, label_mask) in enumerate(zip(image_tensors, label_masks)):
        cam = gradcam(img_tensor.unsqueeze(0), target_class)
        rgb = _denormalize_image(img_tensor, mean, std)
        cam_up = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))

        gt_bin = (np.asarray(label_mask) > 0).astype(np.uint8)
        gt_boundary = cv2.dilate(gt_bin, kernel) & (1 - cv2.erode(gt_bin, kernel))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb)
        ax.imshow(cam_up, cmap='jet', alpha=0.45, vmin=0.0, vmax=1.0)
        ax.contour(gt_boundary, levels=[0.5], colors='white', linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'gradcam_{idx}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return save_dir
