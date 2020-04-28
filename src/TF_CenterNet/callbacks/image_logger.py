import tensorflow as tf
import tensorflow.keras as K
from TF_CenterNet.utils import func_name_scope
from TF_CenterNet.utils import get_input_hw


class ImageLogger(K.callbacks.Callback):
    def __init__(self, cfg, init_op, t_image, t_decode, t_kpt, t_dets):
        super().__init__()
        self.cfg = cfg
        self.writer = tf.summary.FileWriter(cfg.logdir + "/image")
        self.init_op = init_op
        self.summary_op = self.prepare_summary_op(t_image, t_decode, t_kpt, t_dets)

    @func_name_scope("image_logger")
    def prepare_summary_op(self, t_image, t_decode, t_kpt, t_dets):
        t_image = t_image[0:1]
        mask = tf.equal(t_decode["batch_idx"], 0)
        bboxes = tf.boolean_mask(t_decode["bboxes"], mask, axis=0)
        scores = tf.boolean_mask(t_decode["scores"], mask, axis=0)

        h, w = get_input_hw(self.cfg)
        bboxes = tf.stack(
            (bboxes[:, 1] / h, bboxes[:, 0] / w, bboxes[:, 3] / h, bboxes[:, 2] / w,),
            axis=1,
        )

        _, top_k_indices = tf.nn.top_k(scores, 30)
        bboxes = tf.gather(bboxes, top_k_indices)
        
        bboxes = tf.expand_dims(bboxes, 0)
        t_image = tf.image.draw_bounding_boxes(t_image, bboxes)

        t_kpt = tf.transpose(t_kpt[0:1], [3, 1, 2, 0])
        t_pred_kpt = tf.sigmoid(t_dets[0:1, :, :, 4:])
        t_pred_kpt = tf.transpose(t_pred_kpt, [3, 1, 2, 0])
        return tf.summary.merge(
            [
                tf.summary.image("sample_image", t_image, 1, family="sample_image"),
                tf.summary.image("sample_kpt", t_kpt, 6, family="sample_kpt"),
                tf.summary.image(
                    "sample_pred_kpt", t_pred_kpt, 6, family="sample_pred_kpt"
                ),
                tf.summary.tensor_summary("bboxes", bboxes, family="bboxes"),
            ],
        )

    def on_epoch_end(self, epoch, logs={}):
        sess = K.backend.get_session()

        sess.run(self.init_op)
        summary = sess.run(self.summary_op)
        self.writer.add_summary(summary, epoch)
