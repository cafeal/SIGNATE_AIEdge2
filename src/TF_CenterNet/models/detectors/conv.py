import tensorflow as tf
import tensorflow.keras as K
from TF_CenterNet.models.detectors.general_detector import (
    detector,
    _conv2d,
    _relu,
    _batchnorm,
)
from TF_CenterNet.utils import func_name_scope


def block(x, planes, i):
    x = _conv2d(planes, 3, 1, 1)(x)
    x = _batchnorm()(x)
    x = _relu()(x)
    x = _conv2d(planes, 3, 1, 1)(x)
    x = _batchnorm()(x)
    x = _relu()(x)
    return x


@func_name_scope("conv")
def conv(inputs, num_classes):
    x = detector(inputs, num_classes, block)
    return x
