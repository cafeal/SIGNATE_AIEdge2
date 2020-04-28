import tensorflow as tf
import tensorflow.keras as K


def _conv2d(
    planes,
    ksize,
    stride=1,
    dilation=1,
    kernel_initializer=K.initializers.TruncatedNormal(stddev=0.001),
    bias_initializer=None,
    name=None,
):
    return K.layers.Conv2D(
        planes,
        ksize,
        stride,
        "same",
        use_bias=(bias_initializer is not None),
        dilation_rate=dilation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name,
    )


def _batchnorm(name=None):
    return K.layers.BatchNormalization(name=name)


def _relu(name=None):
    return K.layers.ReLU(name=name)


def detector(inputs, num_classes, block):
    if len(inputs) == 1:
        x = inputs[0]
    else:
        planes_list = [int(inp.shape[3]) for inp in inputs[1:]]
        planes_list.append(planes_list[-1])

        x = inputs[0]
        for i, (inp, planes) in enumerate(zip(inputs, planes_list)):
            if i > 0:
                x = K.layers.UpSampling2D()(x)
                x = K.layers.Concatenate()([x, inp])

            x = block(x, planes, i)

    x = _conv2d(64, 3)(x)
    x = _batchnorm()(x)
    x = _relu()(x)
    x = _conv2d(
        num_classes,
        1,
        kernel_initializer=K.initializers.RandomUniform(minval=-0.001, maxval=0.001),
        bias_initializer=K.initializers.Constant(-2.19),
        name="output",
    )(x)

    return x
