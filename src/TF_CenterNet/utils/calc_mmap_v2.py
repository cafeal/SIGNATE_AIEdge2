import numpy as np


def calc_mmap(pred, gt, threshold, max_target_labels=100):
    score = 0
    for _pred, _gt in zip(pred, gt):
        score += calc_map(_pred, _gt, threshold, max_target_labels)
    return score / len(gt)


def calc_map(pr, gt, threshold, max_target_labels):
    aps = []
    for categ in gt:
        ap = 0
        if categ in pr:
            ap = calc_ap(pr[categ], gt[categ], threshold, max_target_labels)

        aps.append(ap)
    return sum(aps) / len(aps)


def calc_ap(pr_bboxes, gt_bboxes, threshold, max_target_labels):
    pr_bboxes = np.asarray(pr_bboxes)
    gt_bboxes = np.asarray(gt_bboxes)

    pr_bboxes = pr_bboxes[:max_target_labels]

    iou_map = calc_iou_map(pr_bboxes, gt_bboxes)
    is_det = iou_map >= threshold  # .to_sparse()

    n_detected = 0
    ap = 0
    for i in range(len(pr_bboxes)):
        _ious = iou_map[i]
        _det = is_det[i].nonzero()[0]

        if len(_det) == 0:
            continue

        n_detected += 1
        ap += n_detected / (i + 1)

        # remove  matched  gt
        max_idx = _det[_ious[_det].argmax()]
        is_det[:, max_idx] = False

    ap /= min(len(gt_bboxes), max_target_labels)
    return ap


def calc_iou_map(pr, gt):
    max_x1 = np.maximum(pr[:, None, 0], gt[None, :, 0])
    min_x2 = np.minimum(pr[:, None, 2], gt[None, :, 2])
    inter_x = np.clip(min_x2 - max_x1, 0, None)
    del max_x1, min_x2

    max_y1 = np.maximum(pr[:, None, 1], gt[None, :, 1])
    min_y2 = np.minimum(pr[:, None, 3], gt[None, :, 3])
    inter_y = np.clip(min_y2 - max_y1, 0, None)
    del max_y1, min_y2

    intersec = inter_x * inter_y

    area_pr = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    score = intersec / (area_pr[:, None] + area_gt[None, :] - intersec)

    return score


if __name__ == "__main__":
    pr = [{0: [[0, 0, 10, 10], [0, 0, 20, 20]]}]
    gt = [{0: [[0, 0, 10, 10], [20, 20, 40, 40]]}]
    print(calc_mmap(pr, gt, 0.75))

