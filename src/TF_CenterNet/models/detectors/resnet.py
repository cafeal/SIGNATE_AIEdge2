import tensorflow as tf
import tensorflow.keras as K
from TF_CenterNet.models.detectors.general_detector import (
    detector,
    _conv2d,
    _relu,
    _batchnorm,
)
from TF_CenterNet.utils import func_name_scope


def _basic_block(x, planes, ksize, stride, dilation=1, downsample=True):
    identity = x
    if downsample:
        identity = _conv2d(planes, 1, stride)(identity)
        identity = _batchnorm()(identity)

    x = _conv2d(planes, 3, stride, dilation)(x)
    x = _batchnorm()(x)
    x = _relu()(x)
    x = _conv2d(planes, 3, 1, dilation)(x)
    x = _batchnorm()(x)

    x = K.layers.Add()([identity, x])
    x = _relu()(x)
    return x


def block(x, planes, i):
    with tf.name_scope(f"basic_block_{i}_1"):
        x = _basic_block(x, planes, 3, 1)
    with tf.name_scope(f"basic_block_{i}_2"):
        x = _basic_block(x, planes, 3, 1)
    return x


@func_name_scope("resnet")
def resnet(inputs, num_classes):
    x = detector(inputs, num_classes, block)
    return x
