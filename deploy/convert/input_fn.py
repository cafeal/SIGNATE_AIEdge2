import os
import pathlib
import random
import numpy as np
import cv2


placeholder = os.environ["INPUT_NODE"]

calib_image_dir = pathlib.Path("./calib_images")
calib_batch_size = int(os.environ["BATCH_SIZE"])
calib_random_seed = 0

image_paths = sorted(calib_image_dir.glob("*.jpg"))
rand = random.Random(calib_random_seed)
rand.shuffle(image_paths)

if os.environ["SCALE"] == "4":
    img_shape = 484, 304
    img_padding = 0, 16, 0, 28
elif os.environ["SCALE"] == "8":
    img_shape = 242, 152
    img_padding = 0, 8, 0, 14
else:
    raise


def normalize(img):
    img = (img / 255.0).astype("float32")

    # max_value = 255.0
    # mean = np.array((0.485, 0.456, 0.406)) * max_value
    # std = np.array((0.229, 0.224, 0.225)) * max_value
    # denominator = np.reciprocal(std, dtype=np.float32)

    # img = img.astype(np.float32)
    # img -= mean[None, None, :]
    # img *= denominator[None, None, :]
    return img


def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, img_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = normalize(img)
    img = cv2.copyMakeBorder(img, *img_padding, cv2.BORDER_CONSTANT, value=0)
    return img


def calib_input(iter):
    _image_paths = image_paths[iter * calib_batch_size : (iter + 1) * calib_batch_size]
    images = [read_image(p.as_posix()) for p in _image_paths]
    return {placeholder: images}
