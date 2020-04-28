import numpy as np
import tensorflow.keras as K
from TF_CenterNet.utils import format_dict
from TF_CenterNet.utils import calc_mmap


class Validation(K.callbacks.Callback):
    def __init__(self, model, steps, init_op, t_inputs, t_loss, t_decode):
        self.model = model
        self.steps = steps
        self.init_op = init_op
        self.t_inputs = t_inputs
        self.t_loss = t_loss
        self.t_decode = t_decode

    def on_epoch_end(self, epoch, logs=None):
        sess = K.backend.get_session()
        sess.run(self.init_op)

        losses = []
        mmaps = []
        for i in range(self.steps):
            inputs, loss_val, decode_vals = sess.run(
                (self.t_inputs, self.t_loss, self.t_decode)
            )
            losses.append(loss_val)

            gt = format_dict.dataset_to_dict(inputs["bboxes"], inputs["categories"])
            pr = format_dict.decode_vals_to_dict(decode_vals)
            mmaps.append(calc_mmap(pr, gt, 0.75, 100))
        logs["val_loss"] = np.mean(losses)
        logs["val_mmap75"] = np.mean(mmaps)