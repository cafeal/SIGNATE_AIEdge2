import tensorflow as tf
import tensorflow.keras as K

from TF_CenterNet.models.extractor import get_extractor
from TF_CenterNet.models.detector import get_detector
from TF_CenterNet.utils import get_input_hw
from TF_CenterNet.utils import func_name_scope


@func_name_scope("centernet")
def get_centernet(cfg, t_images=[None]):
    h, w = get_input_hw(cfg)

    with tf.name_scope("extractor"):
        extractor = get_extractor(cfg)

    input_shapes = [tuple(fmap.shape[1:]) for fmap in extractor.outputs]
    with tf.name_scope("detector"):
        detector = get_detector(cfg, input_shapes)

    models = []
    for t_image in t_images:
        if t_image is None:
            images = K.Input((h, w, 3), name="image")
        else:
            images = K.Input((h, w, 3), tensor=t_image, name="image")
        fmaps = extractor(images)
        dets = detector(fmaps)

        assert h // cfg.stride == dets.shape[1]
        assert w // cfg.stride == dets.shape[2]

        model = K.Model(inputs=images, outputs=dets)
        models.append(model)
    if len(models) == 1:
        return models[0]
    else:
        return models

