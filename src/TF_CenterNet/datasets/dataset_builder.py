import albumentations as albu
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data import Iterator
from torch.utils.data.dataset import Subset
from TF_CenterNet.datasets.aiedge_dataset import AIEdgeDataset
from TF_CenterNet.datasets.aiedge_dataset import AIEdgeTestDataset
from TF_CenterNet.datasets.transform_dataset import TransformDataset
from TF_CenterNet.transforms import PadConstant
from TF_CenterNet.datasets.draw_keypoints import draw_keypoints
from TF_CenterNet.datasets.common_transforms import get_common_transforms
from TF_CenterNet.utils import get_input_hw
from TF_CenterNet.utils import get_fmap_hw
from TF_CenterNet.utils import func_name_scope


class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

        self.dtypes = {
            "image": tf.float32,
            "bboxes": tf.float32,
            "categories": tf.int32,
            "kpt": tf.float32,
            "ct": tf.int32,
            "wh": tf.float32,
            "offset": tf.float32,
            "file": tf.string,
        }

        ih, iw = get_input_hw(cfg)
        fh, fw = get_fmap_hw(cfg)
        self.shapes = {
            "image": (ih, iw, 3),
            "bboxes": (None, 4),
            "categories": (None,),
            "kpt": (fh, fw, 6),
            "ct": (None, 2),
            "wh": (None, 2),
            "offset": (None, 2),
            "file": (),
        }

        self.pad_values = {
            "image": -np.inf,
            "bboxes": -np.inf,
            "categories": -1,
            "kpt": -np.inf,
            "ct": 0,
            "wh": -np.inf,
            "offset": -np.inf,
            "file": "",
        }

    @func_name_scope("dataset")
    def get_dataset(self, n_subset=-1):
        cfg = self.cfg

        train_dataset, valid_dataset = self.get_torch_dataset()

        if n_subset > 0:
            train_dataset = Subset(train_dataset, list(range(100)))
            valid_dataset = Subset(valid_dataset, list(range(100)))

        n_train_iteration = len(train_dataset) // cfg.batch_size
        n_valid_iteration = len(valid_dataset) // cfg.batch_size

        train_dataset = self.to_tf_dataset(train_dataset, shuffle=True)
        valid_dataset = self.to_tf_dataset(valid_dataset, shuffle=False)

        iterator = Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes
        )

        train_init_op = iterator.make_initializer(train_dataset)
        valid_init_op = iterator.make_initializer(valid_dataset)
        input_tensor = iterator.get_next()

        return (
            input_tensor,
            (train_init_op, valid_init_op),
            (n_train_iteration, n_valid_iteration),
        )

    @func_name_scope("test_dataset")
    def get_test_dataset(self):
        dataset = self.get_torch_test_dataset()
        dataset = self.to_tf_test_dataset(dataset)
        iterator = dataset.make_initializable_iterator()

        init_op = iterator.initializer
        input_tensor = iterator.get_next()

        return input_tensor, init_op

    def get_common_transforms(self):
        cfg = self.cfg
        return get_common_transforms(
            (cfg.img_height, cfg.img_width),
            cfg.scale_ratio,
            (cfg.pad_height, cfg.pad_width),
        )

    def get_torch_dataset(self):
        cfg = self.cfg

        augs = []
        if cfg.aug_flip:
            augs.append(albu.HorizontalFlip(p=1.0))
        if cfg.aug_brightness:
            augs.append(albu.RandomBrightness(limit=0.2, p=1.0))
        if cfg.aug_randomcrop:
            aug = albu.RandomSizedBBoxSafeCrop(cfg.img_height, cfg.img_width)
            augs.append(aug)

        common_transforms = self.get_common_transforms()

        train_dataset = TransformDataset(
            AIEdgeDataset(cfg.data_dir, "train", cfg.seed),
            albu.Compose(
                transforms=[*augs, *common_transforms],
                bbox_params=AIEdgeDataset.bbox_params,
            ),
        )

        valid_dataset = TransformDataset(
            AIEdgeDataset(cfg.data_dir, "valid", cfg.seed),
            albu.Compose(
                transforms=common_transforms, bbox_params=AIEdgeDataset.bbox_params,
            ),
        )
        return train_dataset, valid_dataset

    def get_torch_test_dataset(self):
        cfg = self.cfg

        test_dataset = TransformDataset(
            AIEdgeTestDataset(cfg.data_dir),
            albu.Compose(transforms=self.get_common_transforms()),
        )
        return test_dataset

    def to_tf_dataset(self, dataset, shuffle):
        cfg = self.cfg

        keys = {"image", "bboxes", "categories"}
        _dataset = Dataset.from_generator(
            lambda: iter(dataset),
            {k: v for k, v in self.dtypes.items() if k in keys},
            {k: v for k, v in self.shapes.items() if k in keys},
        )

        _dataset = _dataset.map(self.add_features, cfg.num_workers)

        if shuffle and cfg.shuffle_buffer_size > 0:
            _dataset = _dataset.shuffle(
                buffer_size=cfg.shuffle_buffer_size,
                seed=cfg.seed,
                reshuffle_each_iteration=cfg.reshuffle_each_iteration,
            )

        if cfg.batch_size > 1:
            keys = {"image", "bboxes", "categories", "kpt", "ct", "wh", "offset"}
            _dataset = _dataset.padded_batch(
                cfg.batch_size * cfg.n_gpus,
                {k: v for k, v in self.shapes.items() if k in keys},
                {k: v for k, v in self.pad_values.items() if k in keys},
                drop_remainder=True,
            )
        return _dataset

    def to_tf_test_dataset(self, dataset):
        cfg = self.cfg

        keys = {"image", "file"}
        _dataset = Dataset.from_generator(
            lambda: iter(dataset),
            {k: v for k, v in self.dtypes.items() if k in keys},
            {k: v for k, v in self.shapes.items() if k in keys},
        )

        if cfg.batch_size > 1:
            _dataset = _dataset.batch(cfg.batch_size, drop_remainder=False,)
        return _dataset

    @func_name_scope("add_features")
    def add_features(self, inputs):
        cfg = self.cfg

        bboxes = inputs["bboxes"]
        categs = inputs["categories"]
        wh = (bboxes[:, 2:] - bboxes[:, :2]) / cfg.stride
        actural_ct = (bboxes[:, 2:] + bboxes[:, :2]) / 2 / cfg.stride
        ct = tf.round(actural_ct)
        offset = actural_ct - ct

        fh, fw = get_fmap_hw(cfg)
        kpt = draw_keypoints(ct, wh, categs, fh, fw, cfg.n_classes)

        outputs = {
            **inputs,
            "ct": tf.cast(ct, tf.int32),
            "wh": wh,
            "offset": offset,
            "kpt": kpt,
        }

        return outputs
