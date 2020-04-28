import tensorflow.keras as K
from TF_CenterNet.utils import func_name_scope


@func_name_scope("resnet_50")
def resnet_50(t_inputs, input_shape, depth=4):
    backbone = K.applications.ResNet50(
        input_shape=input_shape, include_top=False, input_tensor=t_inputs,
    )

    x = (backbone.get_layer("activation_9").output,)

    if depth >= 2:
        x = (backbone.get_layer("activation_21").output,) + x

    if depth >= 3:
        x = (backbone.get_layer("activation_39").output,) + x

    if depth >= 4:
        x = (backbone.get_layer("activation_48").output,) + x

    return x
