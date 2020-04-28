import tensorflow.keras as K
from TF_CenterNet.utils import func_name_scope


@func_name_scope("mobilenet_v2")
def mobilenet_v2(t_inputs, input_shape):
    backbone = K.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, input_tensor=t_inputs,
    )

    x = (
        backbone.get_layer("out_relu").input,
        backbone.get_layer("block_13_expand_relu").input,
        backbone.get_layer("block_6_expand_relu").input,
        backbone.get_layer("block_3_expand_relu").input,
    )
    return x
