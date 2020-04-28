def get_input_hw(cfg, pad=True):
    pad_height = cfg.pad_height if pad else 0
    pad_width = cfg.pad_width if pad else 0
    h = cfg.img_height // cfg.scale_ratio + pad_height
    w = cfg.img_width // cfg.scale_ratio + pad_width
    return h, w


def get_fmap_hw(cfg):
    h, w = get_input_hw(cfg)
    return h // cfg.stride, w // cfg.stride
