import json

import tensorflow as tf
import tensorflow.keras as K

from TF_CenterNet.utils import format_dict


class Profiling(K.callbacks.Callback):
    def __init__(self, cfg, t_dets, init_op):
        self.cfg = cfg
        self.t_dets = t_dets
        self.init_op = init_op

    def on_train_begin(self, logs=None):
        sess = K.backend.get_session()

        sess.run(self.init_op)
        for _ in range(10):
            sess.run(self.t_dets)

        profiler = tf.profiler.Profiler(sess.graph)
        Builder = tf.profiler.ProfileOptionBuilder

        run_meta = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        sess.run(self.t_dets, options=options, run_metadata=run_meta)

        step = 0
        profiler.add_step(step, run_meta)
        profiler.profile_graph(
            options=(
                Builder(Builder.time_and_memory())
                .with_timeline_output(self.cfg.profiling_file.format(**self.cfg))
                .with_step(step)
                .build()
            )
        )
