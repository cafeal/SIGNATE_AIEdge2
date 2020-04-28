from albumentations import DualTransform
from albumentations.augmentations import functional as F
from albumentations.augmentations.bbox_utils import normalize_bbox, denormalize_bbox
import cv2


class PadConstant(DualTransform):
    """Pad side of the image / max if side is less than desired number.
    Args:
        top (int): top padding.
        bottom (int): bottom padding.
        left (int): left padding.
        right (int): right padding.
        value (int, float, list of int, lisft of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    lisft of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image, mask, bbox, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        top=0,
        bottom=0,
        left=0,
        right=0,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        super(PadConstant, self).__init__(always_apply, p)
        self.pad_top = top
        self.pad_bottom = bottom
        self.pad_left = left
        self.pad_right = right
        self.border_mode = cv2.BORDER_CONSTANT
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadConstant, self).update_params(params, **kwargs)

        params.update(
            {
                "pad_top": self.pad_top,
                "pad_bottom": self.pad_bottom,
                "pad_left": self.pad_left,
                "pad_right": self.pad_right,
            }
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params
    ):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        rows=0,
        cols=0,
        **params
    ):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(
            bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right
        )

    def apply_to_keypoint(
        self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params
    ):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return ("top", "bottom", "left", "right", "border_mode", "value", "mask_value")
