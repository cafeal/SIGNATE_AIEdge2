import tensorflow as tf
from TF_CenterNet.utils import func_name_scope


@func_name_scope("FocalLoss")
def FocalLoss(y_true, y_pred, beta=4, gamma=2):
    eps = 1e-4
    pos_mask = y_true >= 1.0 - eps
    neg_mask = tf.logical_not(pos_mask)
    n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=(1, 2, 3))

    pos_loss = tf.where(
        pos_mask, -tf.pow(1 - y_pred, gamma) * tf.log(y_pred), tf.zeros_like(y_pred),
    )
    pos_loss = tf.reduce_sum(pos_loss, (1, 2, 3)) / n_pos

    neg_loss = tf.where(
        neg_mask,
        -tf.pow(1 - y_true, beta) * tf.pow(y_pred, gamma) * tf.log(1 - y_pred),
        tf.zeros_like(y_pred),
    )
    neg_loss = tf.reduce_sum(neg_loss, (1, 2, 3)) / n_pos
    loss = pos_loss + neg_loss

    return loss
