import numpy as np
import cv2

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
        'kappa': kappa,
        'f1_per_class': f1,
        'precision_per_class': precision,
        'recall_per_class': recall
    }


def zhang_suen_thinning(mask):
    img = np.pad(mask > 0, 1, mode='constant')
    while True:
        changed = False
        for step in range(2):
            p2 = np.roll(img, -1, axis=1)
            p4 = np.roll(img, 1, axis=0)
            p6 = np.roll(img, 1, axis=1)
            p8 = np.roll(img, -1, axis=0)
            p3 = np.roll(p2, 1, axis=0)
            p5 = np.roll(p4, 1, axis=1)
            p7 = np.roll(p6, -1, axis=0)
            p9 = np.roll(p8, -1, axis=1)
            neighbors = (p2.astype(np.uint8) + p3.astype(np.uint8) + p4.astype(np.uint8)
                         + p5.astype(np.uint8) + p6.astype(np.uint8) + p7.astype(np.uint8)
                         + p8.astype(np.uint8) + p9.astype(np.uint8))
            transitions = ((~p2 & p3).astype(np.uint8) + (~p3 & p4).astype(np.uint8)
                           + (~p4 & p5).astype(np.uint8) + (~p5 & p6).astype(np.uint8)
                           + (~p6 & p7).astype(np.uint8) + (~p7 & p8).astype(np.uint8)
                           + (~p8 & p9).astype(np.uint8) + (~p9 & p2).astype(np.uint8))
            if step == 0:
                cond1 = p2 & p4 & p6
                cond2 = p4 & p6 & p8
            else:
                cond1 = p2 & p4 & p8
                cond2 = p2 & p6 & p8
            remove = img & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & ~cond1 & ~cond2
            if remove.any():
                img[remove] = False
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1]


def calc_boundary_iou(pred_labels, gt_labels, boundary_width=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * boundary_width + 1, 2 * boundary_width + 1))
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for pred_label, gt_label in zip(pred_labels, gt_labels):
        gt_bin = (gt_label > 0).astype(np.uint8)
        pred_bin = (pred_label > 0).astype(np.uint8)
        gt_boundary = cv2.dilate(gt_bin, kernel) & (1 - cv2.erode(gt_bin, kernel))
        band = cv2.dilate(gt_boundary, kernel)
        total_tp += int(((pred_bin & gt_bin) & band).sum())
        total_fp += int((pred_bin & (1 - gt_bin) & band).sum())
        total_fn += int(((1 - pred_bin) & gt_bin & band).sum())
    if total_tp + total_fp + total_fn == 0:
        return np.nan
    return total_tp / (total_tp + total_fp + total_fn)


def calc_cldice(pred_labels, gt_labels):
    total_tprec = 0
    total_tsens = 0
    total_pred_skel = 0
    total_gt_skel = 0
    for pred_label, gt_label in zip(pred_labels, gt_labels):
        pred_bin = pred_label > 0
        gt_bin = gt_label > 0
        pred_skel = zhang_suen_thinning(pred_bin)
        gt_skel = zhang_suen_thinning(gt_bin)
        total_tprec += int((pred_skel & gt_bin).sum())
        total_tsens += int((gt_skel & pred_bin).sum())
        total_pred_skel += int(pred_skel.sum())
        total_gt_skel += int(gt_skel.sum())
    tprec = total_tprec / total_pred_skel if total_pred_skel else 0.0
    tsens = total_tsens / total_gt_skel if total_gt_skel else 0.0
    if tprec + tsens == 0:
        return 0.0
    return 2 * tprec * tsens / (tprec + tsens)
