import tensorflow.keras as K


class InitIterator(K.callbacks.Callback):
    def __init__(self, init_op):
        super().__init__()
        self.init_op = init_op

    def on_epoch_begin(self, epoch, logs=None):
        K.backend.get_session().run(self.init_op)
