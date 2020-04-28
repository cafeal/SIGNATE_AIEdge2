import pathlib
import json

import albumentations as albu
import cv2
import pandas as pd
import torch
from torch.utils import data


class AIEdgeDataset(data.Dataset):
    categ2id = {
        "Car": 0,
        "Truck": 1,
        "Bicycle": 2,
        "Pedestrian": 3,
        "Signal": 4,
        "Signs": 5,
        "Bus": -1,
        "SVehicle": -1,
        "Motorbike": -1,
        "Train": -1,
    }
    id2categ = {v: k if v >= 0 else "Other" for k, v in categ2id.items()}

    bbox_params = albu.BboxParams("pascal_voc", label_fields=["categories"])

    def __init__(self, data_dir, split, random_state=None):
        assert split in ["train", "valid"]

        data_path = pathlib.Path(data_dir)
        assert data_path.exists()

        self.anno_path = data_path / "dtc_train_annotations"
        self.image_path = data_path / "dtc_train_images"
        assert self.anno_path.exists()
        assert self.image_path.exists()

        self.split = pd.read_csv(data_path / f"{split}_split.csv")
        if random_state is not None:
            self.split = self.split.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)

        annotations = []
        for file in self.split.file:
            p = self.anno_path / f"{file}.json"
            assert p.exists()
            with open(p) as f:
                a = json.load(f)

            bboxes = []
            categs = []
            for x in a["labels"]:
                b, c = x["box2d"], x["category"]
                bboxes.append((b["x1"], b["y1"], b["x2"], b["y2"]))
                categs.append(self.categ2id[c])

            annotations.append(
                {
                    "file": p.stem,
                    "bboxes": bboxes,
                    "categories": categs,
                    "attributes": a["attributes"],
                    "frameIndex": a["frameIndex"],
                }
            )
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        anno = self.annotations[i]
        file = anno["file"]
        impath = self.image_path / f"{file}.jpg"
        assert impath.exists()
        image = cv2.imread(impath.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ret = {
            "image": image,
            "bboxes": anno["bboxes"],
            "categories": anno["categories"],
        }

        return ret

    @classmethod
    def collate_fn(cls, batch):
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

        return {
            "images": images,
            "bboxes": bboxes,
            "categories": categories,
        }


class AIEdgeTestDataset(data.Dataset):
    categ2id = AIEdgeDataset.categ2id
    id2categ = AIEdgeDataset.id2categ

    def __init__(self, data_dir):
        data_path = pathlib.Path(data_dir)
        assert data_path.exists()

        self.image_path = data_path / "dtc_test_images"
        assert self.image_path.exists()

        self.files = sorted(map(lambda x: x.name, self.image_path.glob("*.jpg")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        impath = self.image_path / file
        assert impath.exists()
        image = cv2.imread(impath.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"image": image, "file": file}

    @classmethod
    def collate_fn(cls, batch):
        images = []
        files = []
        for i, sample in enumerate(batch):
            images.append(sample["image"])
            files.append(sample["file"])

        images = torch.stack(images)
        return {"images": images, "files": files}
