import numpy as np

IGNORE_LABEL = 255  
def calc_semantic_segmentation_confusion(pred_labels, gt_labels, n_class, ignore_label=IGNORE_LABEL):

    if len(pred_labels) != len(gt_labels):
        raise ValueError("Number of predictions and ground truths must be equal.")

    confusion = np.zeros((n_class, n_class), dtype=np.int64)

    for pred_label, gt_label in zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same.')

        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()


        valid_mask = (gt_label != ignore_label)
        valid_gt = gt_label[valid_mask]
        valid_pred = pred_label[valid_mask]

        if valid_gt.size == 0:
            continue  

        if valid_gt.min() < 0 or valid_gt.max() >= n_class:
            raise ValueError(f"Ground truth label out of range [0, {n_class}). "
                             f"Found values: min={valid_gt.min()}, max={valid_gt.max()}")
        if valid_pred.min() < 0 or valid_pred.max() >= n_class:
            raise ValueError(f"Prediction label out of range [0, {n_class}). "
                             f"Found values: min={valid_pred.min()}, max={valid_pred.max()}")

        indices = n_class * valid_gt.astype(int) + valid_pred.astype(int)
        confusion += np.bincount(indices, minlength=n_class**2).reshape(n_class, n_class)

    return confusion


def calc_semantic_segmentation_iou(confusion):
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    iou = np.divide(np.diag(confusion), iou_denominator,
                    out=np.full_like(iou_denominator, np.nan, dtype=np.float64),
                    where=(iou_denominator != 0))
    return iou


def calc_semantic_segmentation_f1(confusion):
    precision = np.divide(np.diag(confusion), confusion.sum(axis=0),
                          out=np.full(confusion.shape[0], np.nan, dtype=np.float64),
                          where=(confusion.sum(axis=0) != 0))
    recall = np.divide(np.diag(confusion), confusion.sum(axis=1),
                       out=np.full(confusion.shape[0], np.nan, dtype=np.float64),
                       where=(confusion.sum(axis=1) != 0))
    f1 = 2 * np.divide(precision * recall, precision + recall,
                       out=np.full_like(precision, np.nan),
                       where=(precision + recall != 0))
    return f1


def calc_semantic_segmentation_precision(confusion):
    return np.divide(np.diag(confusion), confusion.sum(axis=0),
                     out=np.full(confusion.shape[0], np.nan, dtype=np.float64),
                     where=(confusion.sum(axis=0) != 0))


def calc_semantic_segmentation_recall(confusion):
    return np.divide(np.diag(confusion), confusion.sum(axis=1),
                     out=np.full(confusion.shape[0], np.nan, dtype=np.float64),
                     where=(confusion.sum(axis=1) != 0))


def calc_semantic_segmentation_kappa(confusion):
    total = confusion.sum()
    if total == 0:
        return np.nan
    observed_acc = np.trace(confusion) / total
    expected_acc = (confusion.sum(axis=0) * confusion.sum(axis=1)).sum() / (total ** 2)
    kappa = (observed_acc - expected_acc) / (1 - expected_acc) if expected_acc != 1 else np.nan
    return kappa


def eval_semantic_segmentation(pred_labels, gt_labels, n_class, ignore_label=IGNORE_LABEL):

    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels, n_class, ignore_label
    )

    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum() if confusion.sum() > 0 else np.nan
    class_accuracy = np.divide(np.diag(confusion), confusion.sum(axis=1),
                               out=np.full(n_class, np.nan, dtype=np.float64),
                               where=(confusion.sum(axis=1) != 0))

    f1 = calc_semantic_segmentation_f1(confusion)
    precision = calc_semantic_segmentation_precision(confusion)
    recall = calc_semantic_segmentation_recall(confusion)
    kappa = calc_semantic_segmentation_kappa(confusion)

    return {
        'confusion_matrix': confusion,
        'iou': iou,
        'miou': np.nanmean(iou),
        'pixel_accuracy': pixel_accuracy,
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': np.nanmean(class_accuracy),
        'f1': np.nanmean(f1),
        'precision': np.nanmean(precision),
        'recall': np.nanmean(recall),
        'kappa': kappa
    }


def eval_binary_segmentation(pred_labels, gt_labels, ignore_label=255):

    total_intersection = 0
    total_union = 0
    total_correct = 0
    total_pixels = 0
    
    for pred, true in zip(pred_labels, gt_labels):
        mask = (true != ignore_label)
        if mask.sum() == 0:
            continue
            
        pred_masked = pred[mask]
        true_masked = true[mask]
        
        intersection = ((pred_masked == 1) & (true_masked == 1)).sum()
        union = ((pred_masked == 1) | (true_masked == 1)).sum()
        
        total_intersection += intersection
        total_union += union
        
        total_correct += (pred_masked == true_masked).sum()
        total_pixels += mask.sum()
    
    miou = total_intersection / total_union if total_union > 0 else 0
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0
    
    return {
        'miou': float(miou),
        'pixel_accuracy': float(pixel_acc)
    }