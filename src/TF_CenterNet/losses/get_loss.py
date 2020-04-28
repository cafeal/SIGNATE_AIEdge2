import tensorflow as tf
import tensorflow.keras as K

from TF_CenterNet.losses import FocalLoss
from TF_CenterNet.summary_keys import SummaryKeys
from TF_CenterNet.utils import get_input_hw
from TF_CenterNet.utils import get_fmap_hw
from TF_CenterNet.utils import func_name_scope


def get_center_idx(ct):
    B = tf.size(ct[:, 0, 0])
    N = tf.size(ct[0, :, 0])

    batch_idx = tf.range(B)
    batch_idx = tf.broadcast_to(batch_idx[:, None], (B, N))
    ct_yx = tf.gather(ct, [1, 0], axis=2)
    ct_idx = tf.concat((batch_idx[:, :, None], ct_yx), axis=2)
    return ct_idx


def get_loss_of_centers_in_mae(pred_x, x, ct, categs, activ=tf.identity):
    ct_idx = get_center_idx(ct)
    pred_x = tf.gather_nd(pred_x, ct_idx)
    pred_x = activ(pred_x)

    valid_idx = tf.where(categs >= 0)
    x = tf.gather_nd(x, valid_idx)
    pred_x = tf.gather_nd(pred_x, valid_idx)
    batch_segment = tf.gather_nd(ct_idx[:, :, 0], valid_idx)

    ae = tf.abs(pred_x - x)
    mae = tf.segment_mean(ae, batch_segment)
    mae = tf.reduce_mean(mae, axis=1)

    return mae


@func_name_scope("wh_loss")
def get_wh_loss(fmap, wh, ct, categs):
    pred_wh = fmap[:, :, :, :2]
    loss = get_loss_of_centers_in_mae(pred_wh, wh, ct, categs, tf.exp)
    return loss


@func_name_scope("offset_loss")
def get_offset_loss(fmap, offset, ct, categs):
    pred_offset = fmap[:, :, :, 2:4]
    loss = get_loss_of_centers_in_mae(pred_offset, offset, ct, categs, tf.tanh)
    return loss


@func_name_scope("kpt_loss")
def get_kpt_loss(fmap, kpt):
    pred_kpt = fmap[:, :, :, 4:]
    pred_kpt = tf.sigmoid(pred_kpt)
    loss_kpt = FocalLoss(kpt, pred_kpt)
    return loss_kpt
