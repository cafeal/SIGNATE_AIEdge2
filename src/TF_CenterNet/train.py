import hydra
import sys

import albumentations as albu
import tensorflow as tf
import tensorflow.keras as K
from fastprogress.fastprogress import master_bar, progress_bar

from TF_CenterNet.models import get_centernet
from TF_CenterNet.datasets import DatasetBuilder
from TF_CenterNet.losses import get_kpt_loss
from TF_CenterNet.losses import get_wh_loss
from TF_CenterNet.losses import get_offset_loss
from TF_CenterNet.summary_keys import SummaryKeys


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def training(self, sess, writer, epoch):
        sess.run(self.train_init_op)

        ops = {
            "optim": self.optim_op,
            "loss": self.t_loss,
        }

        for i in progress_bar(range(self.n_train_iter), parent=self.mb):
            # ops.update(self._get_summary_ops(i))
            ops["summary"] = self._get_summary_ops(i)

            feed_dict = {self.ph_iter: i, self.ph_epoch: epoch}
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

            try:
                ret = sess.run(ops, feed_dict, run_options)

                writer.add_summary(ret["summary"], self.global_step)
                self.global_step += 1

                self.mb.child.comment = "training step, loss: {loss:.4f}".format(
                    loss=ret["loss"]
                )
            except tf.errors.OutOfRangeError:
                break

    def validation(self, sess, writer):
        sess.run(self.valid_init_op)

        for i in progress_bar(range(self.n_valid_iter), parent=self.mb):
            try:
                summary, loss_val = sess.run((self.valid_summary_op, self.t_loss))
                writer.add_summary(summary, self.global_step)

                self.mb.child.comment = "validation step"

            except tf.errors.OutOfRangeError:
                break

    def _get_summary_ops(self, iteration):
        keys = [SummaryKeys.PER_ITER]

        if iteration == 0:
            keys.append(SummaryKeys.PER_EPOCH)
        if iteration % 100 == 0:
            keys.append(SummaryKeys.PER_100ITER)

        # so = self.summary_ops
        # ops = {k: so[k] for k in keys if so[k] is not None}
        ops = [self.summary_ops[k] for k in keys]
        ops = tf.summary.merge([op for op in ops if op is not None])
        return ops

    def get_optimizer(self):
        cfg = self.cfg.optimizer

        if cfg.name == "sgd":
            return tf.train.MomentumOptimizer(
                cfg.lr, cfg.momentum, use_nesterov=cfg.nesterov
            )
        elif cfg.name == "adam":
            return tf.train.AdamOptimizer(cfg.lr, cfg.beta1, cfg.beta2)
        else:
            raise ValueError()

    def initialize_training_loop(self):
        cfg = self.cfg
        graph = tf.Graph()

        with graph.as_default():
            with tf.name_scope("dataset"):
                t_inputs, init_ops, n_iter = DatasetBuilder(cfg).get_dataset()
                train_init_op, valid_init_op = init_ops
                n_train_iter, n_valid_iter = n_iter

            with tf.name_scope("centernet"):
                model = get_centernet(cfg, t_inputs["image"])
                fmaps = model.output

            with tf.name_scope("loss"):
                loss_kpt = get_kpt_loss(fmaps, t_inputs["kpt"])
                loss_wh = get_wh_loss(
                    fmaps, t_inputs["wh"], t_inputs["ct"], t_inputs["categories"]
                )
                loss_offset = get_offset_loss(
                    fmaps, t_inputs["offset"], t_inputs["ct"], t_inputs["categories"]
                )
                loss = 1.0 * loss_kpt, 0.1 * loss_wh, 1.0 * loss_offset
                loss = tf.reduce_mean(loss)

            with tf.name_scope("optimizer"):
                optim_op = tf.group(
                    (
                        self.get_optimizer().minimize(loss),
                        tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                    )
                )

            summary_ops, ph_epoch, ph_iter = self._initialize_summary_ops()

        tf.summary.scalar("loss", loss, [SummaryKeys.PER_ITER])
        tf.summary.scalar("loss_kpt", loss_kpt, [SummaryKeys.PER_ITER])
        tf.summary.scalar("loss_wh", loss_wh, [SummaryKeys.PER_ITER])
        tf.summary.scalar("loss_offset", loss_offset, [SummaryKeys.PER_ITER])

        tf.summary.image(
            "sample_image", t_inputs["image"][:1], 1, [SummaryKeys.PER_EPOCH]
        )

        _, h, w, _ = t_inputs["kpt"].shape
        sample_kpt = t_inputs["kpt"][0]
        sample_kpt = tf.reshape(sample_kpt, (h, w, 2, 3))
        sample_kpt = tf.transpose(sample_kpt, [2, 0, 1, 3])
        tf.summary.image(
            "sample_keypoint", sample_kpt, 6, [SummaryKeys.PER_EPOCH],
        )

        sample_pred_kpt = tf.sigmoid(fmaps[0, :, :, 4:])
        sample_pred_kpt = tf.reshape(sample_pred_kpt, (h, w, 2, 3))
        sample_pred_kpt = tf.transpose(sample_pred_kpt, [2, 0, 1, 3])
        tf.summary.image(
            "sample_pred_keypoint", sample_pred_kpt, 6, [SummaryKeys.PER_EPOCH]
        )

        self.graph = graph
        self.t_inputs = t_inputs
        self.t_output = fmaps
        self.t_loss = loss
        self.train_init_op = train_init_op
        self.valid_init_op = valid_init_op
        self.n_train_iter = n_train_iter
        self.n_valid_iter = n_valid_iter
        self.optim_op = optim_op
        self.model = model

        self.ph_epoch = ph_epoch
        self.ph_iter = ph_iter
        self.summary_ops = summary_ops

        self.global_step = 0
        self.mb = master_bar(range(self.cfg.n_epochs))

    def _initialize_summary_ops(self):
        ph_epoch = tf.placeholder(tf.int32, shape=None, name="ph_epoch")
        ph_iter = tf.placeholder(tf.int32, shape=None, name="ph_iteration")
        tf.summary.scalar("epoch", ph_epoch, [SummaryKeys.PER_ITER])
        tf.summary.scalar("iteration", ph_iter, [SummaryKeys.PER_ITER])

        var = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="centernet/detector/conv2d_24/kernel:0"
        )[0]
        tf.summary.histogram("conv2d_24_kernel", var, [SummaryKeys.PER_100ITER])

        summary_ops = {
            SummaryKeys.PER_ITER: tf.summary.merge_all(SummaryKeys.PER_ITER),
            SummaryKeys.PER_100ITER: tf.summary.merge_all(SummaryKeys.PER_100ITER),
            SummaryKeys.PER_EPOCH: tf.summary.merge_all(SummaryKeys.PER_EPOCH),
        }

        return summary_ops, ph_epoch, ph_iter

    def run(self):
        cfg = self.cfg

        self.initialize_training_loop()

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(graph=self.graph, config=config) as sess:

            writer = tf.summary.FileWriter(cfg.logdir, sess.graph)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for epoch in self.mb:
                self.training(sess, writer, epoch)
                # self.validation(sess, writer, epoch)


@hydra.main(config_path="configs/config.yaml")
def main(cfg):
    print("CONFIG")
    print(cfg.pretty())

    cfg.data_dir = hydra.utils.to_absolute_path(cfg.data_dir)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
