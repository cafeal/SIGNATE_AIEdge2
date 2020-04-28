import json

import tensorflow as tf
import tensorflow.keras as K

from TF_CenterNet.utils import format_dict


class MakeSubmission(K.callbacks.Callback):
    def __init__(self, cfg, t_file, t_decode, init_op, thresh):
        self.cfg = cfg
        self.t_file = t_file
        self.t_decode = t_decode
        self.init_op = init_op
        self.best_score = 0
        self.thresh = thresh

    def on_epoch_end(self, epoch, logs=None):
        if self.thresh > logs["val_mmap75"]:
            return

        if self.best_score >= logs["val_mmap75"]:
            return
        self.best_score = logs["val_mmap75"]

        sess = K.backend.get_session()
        sess.run(self.init_op)

        submit = {}
        while True:
            try:
                files, decode_vals = sess.run((self.t_file, self.t_decode))
                dic = format_dict.decode_vals_to_dict(
                    decode_vals, convert_categ_name=True,
                )
                files = [s.decode() for s in files]
                submit.update(dict(zip(files, dic)))
            except tf.errors.OutOfRangeError:
                break

        for file in submit:
            for categ in submit[file]:
                submit[file][categ] = (
                    submit[file][categ][:100]
                    .dot(self.cfg.scale_ratio)
                    .astype(int)
                    .tolist()
                )

        with open(self.cfg.submission_file.format(epoch=epoch, **logs), "w") as f:
            json.dump(submit, f)
