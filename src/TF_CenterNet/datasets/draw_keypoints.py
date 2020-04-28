import tensorflow as tf
from TF_CenterNet.utils import func_name_scope


@func_name_scope("gaussian_radius")
def _gaussian_radius(wh, min_overlap=0.7):
    w, h = wh[:, 0], wh[:, 1]
    hw_plus = h + w
    hw_mul = h * w

    a1 = 1
    b1 = hw_plus
    c1 = hw_mul * (1 - min_overlap) / (1 + min_overlap)
    sq1 = tf.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * hw_plus
    c2 = hw_mul * (1 - min_overlap)
    sq2 = (b2 ** 2 - 4 * a2 * c2) ** 0.5
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * hw_plus
    c3 = hw_mul * (min_overlap - 1)
    sq3 = (b3 ** 2 - 4 * a3 * c3) ** 0.5
    r3 = (b3 + sq3) / 2

    r = tf.stack((r1, r2, r3))
    return tf.reduce_min(r, axis=0)


@func_name_scope("draw_keypoints")
def draw_keypoints(ct, wh, categs, fmap_h, fmap_w, n_classes):
    """
    Args:
        ct: center(x, y), shape=(B, 2)
        wh: width and height of bboxes, shape=(B, 2)
        categs: categories, shape=(B)
        fmap_h: feature map height, int
        fmap_w: feature map width, int
        n_classes: number of classes, int
    Returns:
        kpt: keypoint, shape(n_classes, w, h)
    """

    # (2, w, h)
    xy = tf.meshgrid(
        tf.range(fmap_w, dtype=tf.float32), tf.range(fmap_h, dtype=tf.float32)
    )
    xy = tf.stack(xy)

    # (B, w, h)
    dis = tf.norm(xy[None, :, :, :] - ct[:, :, None, None], axis=1)
    # (B)
    sig = _gaussian_radius(wh)
    # (B, w, h)
    indiv_kpts = tf.exp(-dis / (2 * sig[:, None, None] ** 2))
    false_kpts = tf.zeros_like(indiv_kpts)

    kpts = []
    for i in range(n_classes):
        # (B,)
        mask = tf.equal(categs, i)
        # (B, w, h)
        _kpts = tf.where(mask, indiv_kpts, false_kpts)
        # (w, h)
        kpt = tf.reduce_max(_kpts, axis=0)
        kpts.append(kpt)
    # (w, h, n_classes)
    kpts = tf.stack(kpts, axis=2)
    return kpts
