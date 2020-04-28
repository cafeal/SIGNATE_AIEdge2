import torch


def mmap(pred, gt, threshold, max_target_labels=100):
    score = 0
    for _pred, _gt in zip(pred, gt):
        score += calc_map(_pred, _gt, threshold, max_target_labels)
    return score / len(gt)


def calc_map(pr, gt, threshold, max_target_labels):
    aps = []
    for categ in gt:
        ap = 0
        if categ in pr:
            ap = calc_ap(
                pr[categ][:max_target_labels], gt[categ], threshold, max_target_labels
            )

        aps.append(ap)
    return sum(aps) / len(aps)


def calc_ap(pr_bboxes, gt_bboxes, threshold, max_target_labels):
    pr_bboxes = pr_bboxes[:max_target_labels]

    iou_map = calc_iou_map(pr_bboxes, gt_bboxes)
    is_det = iou_map >= threshold  # .to_sparse()

    n_detected = 0
    ap = 0
    for i in range(len(pr_bboxes)):
        _ious = iou_map[i]
        _det = is_det[i].nonzero().flatten()

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
    max_x1 = (
        torch.stack(
            (pr[:, 0].expand(len(gt), len(pr)).t(), gt[:, 0].expand(len(pr), len(gt)),)
        )
        .max(dim=0)
        .values
    )

    min_x2 = (
        torch.stack(
            (pr[:, 2].expand(len(gt), len(pr)).t(), gt[:, 2].expand(len(pr), len(gt)),)
        )
        .min(dim=0)
        .values
    )

    inter_x = (min_x2 - max_x1).clamp(min=0)
    del max_x1, min_x2

    max_y1 = (
        torch.stack(
            (pr[:, 1].expand(len(gt), len(pr)).t(), gt[:, 1].expand(len(pr), len(gt)),)
        )
        .max(dim=0)
        .values
    )

    min_y2 = (
        torch.stack(
            (pr[:, 3].expand(len(gt), len(pr)).t(), gt[:, 3].expand(len(pr), len(gt)),)
        )
        .min(dim=0)
        .values
    )

    inter_y = (min_y2 - max_y1).clamp(min=0)
    del max_y1, min_y2

    intersec = inter_x * inter_y

    area_pr = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    score = intersec.float() / (area_pr[:, None] + area_gt[None, :] - intersec)

    return score
