import tensorflow as tf
import tensorflow.keras as K
from TF_CenterNet.utils import func_name_scope
from TF_CenterNet.models.detectors.resnet import resnet
from TF_CenterNet.models.detectors.conv import conv

_detectors = {
    "resnet": resnet,
    "conv": conv,
}


@func_name_scope("detector")
def get_detector(cfg, input_shapes, t_inputs=None):
    model = cfg.detector
    if model not in _detectors:
        raise NotImplementedError("{} is not implemented".format(model))

    if t_inputs is None:
        inputs = [K.layers.Input(shape) for shape in input_shapes]
    else:
        inputs = [
            K.layers.Input(shape, tensor=t_inp)
            for shape, t_inp in zip(input_shapes, t_inputs)
        ]

    dets = _detectors[model](inputs, 4 + cfg.n_classes)
    detector = K.Model(inputs=inputs, outputs=dets, name="detector_model")
    return detector
