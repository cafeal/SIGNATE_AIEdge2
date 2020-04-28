from functools import partial
import tensorflow.keras as K
from TF_CenterNet.utils import get_input_hw
from TF_CenterNet.utils import func_name_scope
from TF_CenterNet.models.extractors.mobilenet_v2 import mobilenet_v2
from TF_CenterNet.models.extractors.resnet_50 import resnet_50


_extractors = {
    "mobilenet_v2": mobilenet_v2,
    "resnet_50": resnet_50,
    "resnet_50_light": partial(resnet_50, depth=3),
    "resnet_50_ultralight": partial(resnet_50, depth=2),
    "resnet_50_extralight": partial(resnet_50, depth=1),
}


@func_name_scope("extractor")
def get_extractor(cfg, t_inputs=None):
    model = cfg.extractor
    if model not in _extractors:
        raise NotImplementedError("{} is not implemented".format(model))

    h, w = get_input_hw(cfg)

    if t_inputs is None:
        inputs = K.Input((h, w, 3))
    else:
        inputs = K.Input((h, w, 3), tensor=t_inputs)

    fmaps = _extractors[model](inputs, (h, w, 3))

    extractor = K.Model(inputs=inputs, outputs=fmaps, name="extractor_model")
    return extractor
