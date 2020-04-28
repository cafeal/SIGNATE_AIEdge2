from argparse import Namespace
import json
import os
import pathlib
import random
import time

import hydra

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

import torch
from torch import nn
from torch.utils import data
import torchvision
import pytorch_lightning as pl

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

from datasets import AIEdgeDataset
from datasets import AIEdgeTestDataset
from datasets import TransformDataset
from losses import FocalLoss
from models.extractors import (
    MobileNetV2Extractor,
    ResNetExtractor,
    MNASNetExtractor,
    EfficientNetExtractor,
)
from models.detectors import ResNetDetector, FCNDetector, MobileNetV2Detector
from utils.draw_keypoint import draw_keypoint
from utils.viz_bboxes import viz_bboxes
from utils.decode import decode
from utils.mmap import mmap


class CollateFn:
    def __init__(self, stride, hparams, split):
        assert split in ["train", "valid"]
        self.stride = stride
        self.hparams = hparams
        self.split = split

    def __call__(self, batch):
        stride = self.stride
        hparams = self.hparams
        split = self.split

        images = []
        bboxes = []
        categories = []
        for _, sample in enumerate(batch):
            images.append(sample["image"])
            bboxes.append(sample["bboxes"])
            categories.append(sample["categories"])

        images = torch.stack(images)
        bboxes = [torch.FloatTensor(x) for x in bboxes]
        categories = [torch.LongTensor(x) for x in categories]

        if split == "valid":
            return {
                "images": images,
                "bboxes": bboxes,
                "categories": categories,
            }

        _, _, H, W = images.shape

        kpts = []
        whs = []
        offsets = []
        cts = []
        for bbox, categs in zip(bboxes, categories):
            wh = bbox[:, 2:] - bbox[:, :2]
            ct = bbox[:, :2] + wh / 2

            offset = ct / stride - ct // stride
            ct = (ct // stride).long()
            wh = wh / stride

            kpt = draw_keypoint(
                ct, wh, categs, W // stride, H // stride, hparams.n_classes,
            )

            kpts.append(kpt.detach())
            offsets.append(offset.detach())
            whs.append(wh.detach())
            cts.append(ct.detach())

        kpts = torch.stack(kpts).detach()

        return {
            "images": images,
            "kpts": kpts,
            "offsets": offsets,
            "whs": whs,
            "cts": cts,
        }


def get_network(extractor, detector, n_classes):
    if extractor == "mnasnet_0_5":
        pretrained = torchvision.models.mnasnet0_5(True)
        extractor = MNASNetExtractor.setup(pretrained)
    elif extractor == "mnasnet_1_0":
        pretrained = torchvision.models.mnasnet1_0(True)
        extractor = MNASNetExtractor.setup(pretrained)
    elif extractor == "mobilenet_v2":
        pretrained = torchvision.models.mobilenet_v2(True)
        extractor = MobileNetV2Extractor.setup(pretrained)
    elif extractor == "resnet_18":
        pretrained = torchvision.models.resnet18(True)
        extractor = ResNetExtractor.setup(pretrained)
    elif extractor == "efficientnet_b0":
        extractor = EfficientNetExtractor.get_pretrained("efficientnet-b0")
    elif extractor == "efficientnet_b1":
        extractor = EfficientNetExtractor.get_pretrained("efficientnet-b1")
    elif extractor == "efficientnet_b2":
        extractor = EfficientNetExtractor.get_pretrained("efficientnet-b2")
    else:
        raise

    if detector == "fcn":
        detector = FCNDetector(extractor, 4 + n_classes)
    elif detector == "mobilenet_v2":
        detector = MobileNetV2Detector(extractor, 4 + n_classes)
    elif detector == "resnet":
        detector = ResNetDetector(extractor, 4 + n_classes)
    else:
        raise

    return detector


class CenterNet(pl.LightningModule):
    loss_fn = {
        "kpt": FocalLoss(),
        "offset": nn.L1Loss(),
        "wh": nn.L1Loss(),
    }

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.net = get_network(hparams.extractor, hparams.detector, hparams.n_classes)

        # init dataset
        self.train_dataset = AIEdgeDataset(hparams.data_dir, "train")
        self.valid_dataset = AIEdgeDataset(hparams.data_dir, "valid")
        self.test_dataset = AIEdgeTestDataset(hparams.data_dir)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        kpts = batch["kpts"]
        whs = batch["whs"]
        offsets = batch["offsets"]
        cts = batch["cts"]

        B, _, _, _ = images.shape

        outputs = self.forward(images)

        losses = {
            "kpt": 0,
            "offset": 0,
            "wh": 0,
        }

        for out, kpt, wh, offset, ct in zip(outputs, kpts, whs, offsets, cts):
            pred_offset = out[:2, ct[:, 1], ct[:, 0]].permute(1, 0).tanh()
            pred_wh = out[2:4, ct[:, 1], ct[:, 0]].permute(1, 0).exp()
            pred_kpt = out[4:].sigmoid()

            losses["kpt"] += self.loss_fn["kpt"](pred_kpt, kpt) / B
            losses["offset"] += self.loss_fn["offset"](pred_offset, offset) / B
            losses["wh"] += self.loss_fn["wh"](pred_wh, wh) / B

        loss = 1.0 * losses["kpt"] + 0.1 * losses["offset"] + 1.0 * losses["wh"]

        tensorboard_logs = {"train_loss": loss, **losses}
        return {"loss": loss, "log": tensorboard_logs}

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        super().optimizer_step(
            current_epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            second_order_closure=None,
        )

        self.logger.experiment.add_scalar(
            "lr", optimizer.param_groups[0]["lr"], global_step=self.global_step
        )

    def _decode(self, images):
        outputs = self.forward(images).detach()

        offset = outputs[:, :2].tanh().detach()
        wh = outputs[:, 2:4].exp().detach()
        kpt = outputs[:, 4:].sigmoid().detach()

        bboxes, categs, scores = decode(kpt, wh, offset, 100)
        bboxes *= self.net.stride

        return bboxes, categs, scores

    def _gather_bbox_by_category(self, bboxes, categs):
        result = []
        for _bboxes, _categs in zip(bboxes, categs):
            d = {}
            for cname, cid in AIEdgeDataset.categ2id.items():
                b = _bboxes[_categs == cid]
                if len(b) > 0:
                    d[cname] = b
            result.append(d)
        return result

    def _calc_mmap(
        self,
        gt_bboxes,
        gt_categories,
        pred_bboxes,
        pred_categories,
        threshold,
        max_target_labels,
    ):
        gt = self._gather_bbox_by_category(gt_bboxes, gt_categories)
        pred = self._gather_bbox_by_category(pred_bboxes, pred_categories)
        return mmap(pred, gt, threshold, max_target_labels)

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        gt_bboxes = batch["bboxes"]
        gt_categories = batch["categories"]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        pred_bboxes, pred_categories, scores = self._decode(images)
        end.record()

        val_mmap75 = self._calc_mmap(
            gt_bboxes, gt_categories, pred_bboxes, pred_categories, 0.75, 100,
        )

        # output image with bboxes
        if batch_idx == 0:
            id2categ = AIEdgeDataset.id2categ
            for i in range(5):
                img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                sample_scores = scores[i].detach().cpu().numpy()
                sample_bboxes = pred_bboxes[i].detach().cpu().numpy().astype(int)
                sample_gt_bboxes = gt_bboxes[i].detach().cpu().numpy().astype(int)
                sample_categs = pred_categories[i].detach().cpu().numpy().astype(int)
                sample_gt_categs = gt_categories[i].detach().cpu().numpy().astype(int)

                # convert category id to category name
                sample_categs = [id2categ[x] for x in sample_categs]
                sample_gt_categs = [id2categ[x] for x in sample_gt_categs]

                img = viz_bboxes(
                    img,
                    bboxes=sample_gt_bboxes,
                    categories=sample_gt_categs,
                    color=(10, 255, 10),
                )

                img = viz_bboxes(
                    img,
                    bboxes=sample_bboxes,
                    categories=sample_categs,
                    color=(255, 10, 10),
                    alpha=sample_scores,
                )

                self.logger.experiment.add_image(
                    f"sample{i}", img.transpose(2, 0, 1), global_step=self.global_step,
                )

        torch.cuda.synchronize()
        time_per_batch_size = start.elapsed_time(end) / len(images)

        return {"val_mmap75": val_mmap75, "time_per_batch_size": time_per_batch_size}

    def validation_end(self, outputs):
        avg_mmap = np.mean([x["val_mmap75"] for x in outputs])
        avg_time_per_batch_size = np.mean([x["time_per_batch_size"] for x in outputs])
        ret = {"val_mmap75": avg_mmap, "time_per_batch_size": avg_time_per_batch_size}
        return {
            "log": ret,
            "progress_bar": ret,
            "val_mmap75": avg_mmap,
            "val_loss": avg_mmap,
        }

    def test_step(self, batch, batch_idx):
        images = batch["images"]
        files = batch["files"]

        bboxes, categs, scores = self._decode(images)

        bboxes = bboxes.cpu().numpy().astype(int) * self.hparams.scale_ratio
        scores = scores.cpu().numpy().astype(float)
        categs = categs.cpu().numpy().astype(int)

        submit = self._gather_bbox_by_category(bboxes, categs)
        for s in submit:
            for categ in s:
                s[categ] = s[categ].tolist()
        submit = dict(zip(files, submit))
        return submit

    def test_end(self, outputs):
        submit = {}
        for x in outputs:
            submit.update(x)

        with open(self.hparams.submission_file, "w") as f:
            json.dump(submit, f)

        return {}

    def configure_optimizers(self):
        lr = self.hparams.lr
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr, 0.9)
        else:
            raise

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.hparams.reduce_lr_round,
            cooldown=10,
        )
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        hparams = self.hparams

        augs = []
        if hparams.aug_flip:
            augs.append(albu.HorizontalFlip(p=1.0))
        if hparams.aug_brightness:
            augs.append(albu.RandomBrightness(limit=0.2, p=1.0))
        if hparams.aug_randomcrop:
            aug = albu.RandomSizedBBoxSafeCrop(hparams.img_height, hparams.img_width)
            augs.append(aug)

        dataset = TransformDataset(
            self.train_dataset,
            albu.Compose(
                [
                    *augs,
                    albu.Resize(hparams.img_height // 4, hparams.img_width // 4),
                    albu.ToFloat(255),
                    ToTensorV2(),
                ],
                bbox_params=AIEdgeDataset.bbox_params,
            ),
        )

        if self.hparams.gpus > 1:
            sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None

        return data.DataLoader(
            dataset,
            batch_size=hparams.batch_size,
            collate_fn=CollateFn(self.net.stride, hparams, "train"),
            num_workers=hparams.num_workers,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=(hparams.gpus > 0),
            drop_last=True,
        )

    @pl.data_loader
    def val_dataloader(self):
        hparams = self.hparams

        dataset = TransformDataset(
            self.valid_dataset,
            albu.Compose(
                [
                    albu.Resize(
                        hparams.img_height // hparams.scale_ratio,
                        hparams.img_width // hparams.scale_ratio,
                    ),
                    albu.ToFloat(255),
                    ToTensorV2(),
                ],
                bbox_params=AIEdgeDataset.bbox_params,
            ),
        )

        if hparams.gpus > 1:
            sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None

        return data.DataLoader(
            dataset,
            batch_size=hparams.batch_size,
            collate_fn=CollateFn(self.net.stride, hparams, "valid"),
            num_workers=hparams.num_workers,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=(hparams.gpus > 0),
        )

    @pl.data_loader
    def test_dataloader(self):
        hparams = self.hparams

        dataset = TransformDataset(
            self.test_dataset,
            albu.Compose(
                [
                    albu.Resize(
                        hparams.img_height // hparams.scale_ratio,
                        hparams.img_width // hparams.scale_ratio,
                    ),
                    albu.ToFloat(255),
                    ToTensorV2(),
                ],
            ),
        )

        if self.hparams.gpus > 1:
            sampler = data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = None

        return data.DataLoader(
            dataset,
            batch_size=hparams.batch_size,
            collate_fn=AIEdgeTestDataset.collate_fn,
            num_workers=hparams.num_workers,
            sampler=sampler,
            pin_memory=(hparams.gpus > 0),
        )


@hydra.main(config_path="configs/config.yml")
def main(cfg):
    cfg.data_dir = hydra.utils.to_absolute_path(cfg.data_dir)

    print("CONFIG")
    print(cfg.pretty())

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    hparams = Namespace(**cfg)
    model = CenterNet(hparams)

    trainer = pl.Trainer(
        logger=True,
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        gpus=cfg.gpus,
        val_percent_check=1.0,
        num_sanity_val_steps=5,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.getcwd(), monitor="val_mmap75", mode="max",
        ),
        early_stop_callback=pl.callbacks.EarlyStopping(
            monitor="val_mmap75", patience=cfg.early_stopping_round, mode="max",
        ),
        resume_from_checkpoint=cfg.resume if cfg.resume else None,
        distributed_backend="ddp" if cfg.gpus > 1 else None,
    )
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    main()
