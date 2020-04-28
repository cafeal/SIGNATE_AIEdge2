import torch


def _gaussian_radius(wh, min_overlap=0.7):
    w, h = wh[:, 0], wh[:, 1]
    hw_plus = h + w
    hw_mul = h * w
    
    a1  = 1
    b1  = hw_plus
    c1  = hw_mul * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * hw_plus
    c2  = hw_mul * (1 - min_overlap) 
    sq2 = (b2 ** 2 - 4 * a2 * c2) ** 0.5
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * hw_plus
    c3  = hw_mul * (min_overlap - 1)
    sq3 = (b3 ** 2 - 4 * a3 * c3) ** 0.5
    r3  = (b3 + sq3) / 2

    r = torch.stack((r1, r2, r3))
    return torch.min(r, 0)[0]

def draw_keypoint(ct, wh, categs, im_w, im_h, n_classes):
    """
    Args:
        ct: center(x, y), shape=(B, 2)
        wh: width and height of bboxes, shape=(B, 2)
        categs: categories, shape=(B)
        im_w: image width, int
        im_h: image height, int
        n_classes: number of classes, int
    Returns:
        kpt: keypoint, shape(n_classes, w, h)
    """

    device = ct.device
    
    # (2, w, h)
    index_map = torch.stack(
        torch.meshgrid(
            torch.arange(im_h, dtype=torch.float32),
            torch.arange(im_w, dtype=torch.float32),
        )
    )
    index_map = index_map.to(device)
    
    # (B, w, h)
    dis = torch.norm(index_map[None, :, :, :] - ct[:, [1, 0], None, None].float(), dim=1)
    # (B)
    sig = _gaussian_radius(wh)
    # (B, w, h)
    kpts = torch.exp(-dis / 2*sig[:, None, None]**2)
    
    # (n_classes, w, h)
    kpt = torch.zeros(n_classes, im_h, im_w).to(device)
    for i in range(n_classes):
        if i in categs:
            kpt[i] = kpts[categs==i].max(dim=0)[0]
    return kpt
