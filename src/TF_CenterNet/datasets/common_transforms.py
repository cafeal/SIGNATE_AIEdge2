import albumentations as albu


def get_common_transforms(shape, scale_ratio, pad):
    h, w = shape
    ph, pw = pad

    return [
        albu.Resize(h // scale_ratio, w // scale_ratio),
        # albu.Normalize(),
        albu.ToFloat(255),
        PadConstant(0, cfg.ph, 0, cfg.pw),
    ]