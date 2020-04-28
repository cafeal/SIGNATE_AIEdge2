import tensorflow as tf
import tensorflow.keras as K
from TF_CenterNet.utils import func_name_scope
from TF_CenterNet.utils import get_input_hw


@func_name_scope("docode")
def decode(cfg, dets):
    """Decode centernet output to bounding boxes
    
    Parameters
    ----------
    dets : Tensor, (batch_size, h, w, 4 + n_classes)
        dets[:, :, :, 0]: box width
        dets[:, :, :, 1]: box height
        dets[:, :, :, 2]: offset x
        dets[:, :, :, 3]: offset y
        dets[:, :, :, 4:]: keypoint heat maps
    
    Returns
    -------
    ret : Tensor (?, 7)
        ret[:, 0]: batch index
        ret[:, 1]: categories
        ret[:, 2]: confidence score
        ret[:, 3]: left
        ret[:, 4]: top
        ret[:, 5]: right
        ret[:, 6]: bottom
    """

    kpt = tf.sigmoid(dets[:, :, :, 4:])
    wh = tf.exp(dets[:, :, :, :2])
    offset = tf.tanh(dets[:, :, :, 2:4])

    # nms
    local_peak = K.layers.MaxPool2D(3, 1, "same")(kpt)
    peak_mask = tf.equal(kpt, local_peak)
    thresh_mask = tf.greater_equal(kpt, 0.01)
    mask = tf.logical_and(peak_mask, thresh_mask)

    # [?, 4=(batch_id, y, x, category_id)]
    kpt_idx = tf.where(mask)

    batch_idx = kpt_idx[:, 0]
    ct_idx = kpt_idx[:, :3]
    categs = kpt_idx[:, 3]

    # [?,]
    kpt = tf.gather_nd(kpt, kpt_idx)
    scores = tf.sigmoid(kpt)

    # [?, 2]
    wh = tf.gather_nd(wh, ct_idx)
    # [?, 2]
    offset = tf.gather_nd(offset, ct_idx)

    ct_xy = tf.reverse(ct_idx[:, 1:], axis=[1])
    ct_xy = tf.cast(ct_xy, tf.float32) + offset
    bboxes = tf.concat((ct_xy - wh / 2, ct_xy + wh / 2), 1)
    bboxes = bboxes * cfg.stride

    h, w = get_input_hw(cfg, pad=False)
    bboxes = tf.clip_by_value(bboxes, 0, [[w, h, w, h]])

    ret = {
        "batch_idx": batch_idx,
        "categs": categs,
        "scores": scores,
        "bboxes": bboxes,
    }
    return ret
