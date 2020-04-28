import torch
import torch.nn.functional as F


def _nms(kpt):
    """
    Args:
        kpt: keypoint heatmap (B, C, H, W)
    Returns
        suppressed keypoint, (B, C, H, W)
    """
    B, C, H, W = kpt.shape
    max_neighbor = F.max_pool2d(kpt.view(-1, H, W), 3, 1, 1)
    max_neighbor = max_neighbor.view(B, C, H, W)
    mask = kpt == max_neighbor
    return kpt * mask.float()


def _nms2(kpt):
    tl = kpt[:, :, 1:, 1:] >= kpt[:, :, 0:-1, 0:-1]
    tm = kpt[:, :, 1:, 1:-1] >= kpt[:, :, 0:-1, 1:-1]
    tr = kpt[:, :, 1:, :-1] >= kpt[:, :, 0:-1, 1:]
    mt = kpt[:, :, 1:-1, 1:] >= kpt[:, :, 1:-1, :-1]

    peaks = torch.stack(
        (
            tl[:, :, :-1, :-1],
            ~tl[:, :, 1:, 1:],
            tm[:, :, :-1, :],
            ~tm[:, :, 1:, :],
            tr[:, :, :-1, 1:],
            ~tr[:, :, 1:, :-1],
            mt[:, :, :, :-1],
            ~mt[:, :, :, 1:],
        )
    ).all(axis=0)

    peaks = F.pad(peaks, (1, 1, 1, 1))

    return peaks


def _kpt_indices(kpt, k):
    """
    Args:
        kpt: keypoint heatmap (B, C, H, W)
        k: top-k
    Returns:
        x coordinates (B, topk)
        y coordinates (B, topk)
        categories (B, topk)
    """
    B, C, H, W = kpt.shape
    values, indices = torch.topk(kpt.view(B, -1), k)

    idx_c = indices // (H * W)
    indices = indices % (H * W)
    idx_y = indices // W
    idx_x = indices % W

    return idx_x, idx_y, idx_c


def decode(kpt, wh, offset, topk):
    """
    Args:
        kpt: keypoint heatmap, (B, C, H, W)
        wh: width and height, (B, 2, H, W)
        offset: offset, (B, 2, H, W)
        topk: number of detected bboxes, int
    Returns:
        bounding boxes, (B, topk, 4)
        categories, (B, topk)
        scores, (B, topk)
    """

    B, C, H, W = kpt.shape

    kpt = _nms(kpt)
    idx_x, idx_y, idx_c = _kpt_indices(kpt, topk)
    idx_b = torch.arange(B).repeat(topk, 1).t()

    offset = offset[idx_b, :, idx_y, idx_x]
    wh = wh[idx_b, :, idx_y, idx_x]

    categories = idx_c
    scores = kpt[idx_b, idx_c, idx_y, idx_x]

    ct = torch.stack((idx_x, idx_y), dim=2).float()
    ct += offset

    bbox = torch.cat((ct - wh / 2, ct + wh / 2,), dim=2)

    # clip
    bbox[:, :, [0, 2]] = bbox[:, :, [0, 2]].clamp(0, W)
    bbox[:, :, [1, 3]] = bbox[:, :, [1, 3]].clamp(0, H)

    return bbox, categories, scores
