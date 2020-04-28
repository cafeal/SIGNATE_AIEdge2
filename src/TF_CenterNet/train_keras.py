import hydra
import os
import sys

import numpy as np
import albumentations as albu
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import multi_gpu_model

from TF_CenterNet.datasets import DatasetBuilder
from TF_CenterNet.models import get_centernet
from TF_CenterNet.losses.get_loss import get_kpt_loss, get_wh_loss, get_offset_loss
from TF_CenterNet.summary_keys import SummaryKeys
from TF_CenterNet.utils import decode
from TF_CenterNet.utils import calc_mmap
from TF_CenterNet.utils import format_dict
from TF_CenterNet.utils import func_name_scope
from TF_CenterNet.callbacks import ImageLogger
from TF_CenterNet.callbacks import InitIterator
from TF_CenterNet.callbacks import Validation
from TF_CenterNet.callbacks import MakeSubmission
from TF_CenterNet.callbacks import Profiling


@func_name_scope("optimizer")
def get_optimizer(cfg):
    cfg = cfg.optimizer
    if cfg.name == "adam":
        return K.optimizers.Adam(cfg.lr, cfg.beta1, cfg.beta2, amsgrad=cfg.amsgrad)
    else:
        raise ValueError()


@hydra.main(config_path="configs/config.yaml")
def main(cfg):
    print("CONFIG")
    print(cfg.pretty())

    cfg.data_dir = hydra.utils.to_absolute_path(cfg.data_dir)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    K.backend.set_session(tf.Session(config=config))

    # dataset
    builder = DatasetBuilder(cfg)
    t_inputs, init_ops, n_iter = builder.get_dataset(n_subset=cfg.n_subset)
    init_train_op, init_valid_op = init_ops
    n_train_iter, n_valid_iter = n_iter
    t_test_inputs, test_init_op = builder.get_test_dataset()

    device_name = "/gpu:0" if cfg.n_gpus == 1 else "/cpu:0"
    with tf.device(device_name):
        model, test_model = get_centernet(
            cfg, [t_inputs["image"], t_test_inputs["image"]]
        )
        if cfg.resume:
            resume_path = hydra.utils.to_absolute_path(cfg.resume)
            model.load_weights(resume_path)

    # multi
    if cfg.n_gpus > 1:
        model = multi_gpu_model(model, cfg.n_gpus, False, False)
    dets = model.output
    test_dets = test_model.output

    # decode
    t_decode = decode(cfg, dets)
    t_test_decode = decode(cfg, test_dets)

    # loss
    loss_kpt = get_kpt_loss(dets, t_inputs["kpt"])
    loss_wh = get_wh_loss(dets, t_inputs["wh"], t_inputs["ct"], t_inputs["categories"])
    loss_offset = get_offset_loss(
        dets, t_inputs["offset"], t_inputs["ct"], t_inputs["categories"]
    )
    loss = 1.0 * loss_kpt, 0.1 * loss_wh, 1.0 * loss_offset
    t_loss = tf.reduce_mean(loss)
    model.add_loss(t_loss)

    optimizer = get_optimizer(cfg)
    model.compile(optimizer)

    model.fit(
        steps_per_epoch=n_train_iter,
        epochs=cfg.n_epochs,
        use_multiprocessing=True,
        callbacks=[
            K.callbacks.TerminateOnNaN(),
            Profiling(cfg, test_dets, test_init_op),
            InitIterator(init_train_op),
            Validation(model, n_valid_iter, init_valid_op, t_inputs, t_loss, t_decode),
            ImageLogger(
                cfg, init_valid_op, t_inputs["image"], t_decode, t_inputs["kpt"], dets,
            ),
            MakeSubmission(
                cfg, t_test_inputs["file"], t_test_decode, test_init_op, thresh=0.15,
            ),
            K.callbacks.ModelCheckpoint(
                cfg.logdir + "/model.{epoch:04d}-{val_mmap75:.4f}.h5",
                monitor="val_mmap75",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor="val_mmap75",
                mode="max",
                factor=0.3,
                patience=cfg.reduce_lr_round,
                min_lr=cfg.optimizer.lr * 0.01,
            ),
            K.callbacks.EarlyStopping(
                monitor="val_mmap75",
                mode="max",
                min_delta=0,
                patience=cfg.early_stopping_round,
                verbose=1,
            ),
            K.callbacks.TensorBoard(
                log_dir=cfg.logdir,
                write_graph=True,
                write_grads=False,
                write_images=False,
            ),
        ],
    )


if __name__ == "__main__":
    main()
