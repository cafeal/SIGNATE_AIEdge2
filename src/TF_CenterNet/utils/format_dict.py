import numpy as np
from TF_CenterNet.datasets import AIEdgeDataset


def dataset_to_dict(bboxes, categs):
    dets = []
    for _bboxes, _categs in zip(bboxes, categs):
        det = {}
        for categ_id in np.unique(_categs):
            if categ_id == -1:
                continue
            det[categ_id] = _bboxes[_categs == categ_id]
        dets.append(det)
    return dets


def decode_vals_to_dict(decode_vals, with_score=False, convert_categ_name=False):
    batch_idx = decode_vals["batch_idx"]
    bboxes = decode_vals["bboxes"]
    categs = decode_vals["categs"]
    scores = decode_vals["scores"]

    pr = []
    for batch_id in np.unique(batch_idx):
        batch_mask = batch_idx == batch_id
        in_batch_categs = categs[batch_mask]

        det = {}
        for categ_id in np.unique(in_batch_categs):
            categ_mask = categs == categ_id
            mask = batch_mask & categ_mask

            _bboxes = bboxes[mask]
            _scores = scores[mask]

            order = np.argsort(_scores)[::-1]
            _bboxes = _bboxes[order]
            _scores = _scores[order]

            if convert_categ_name:
                key = AIEdgeDataset.id2categ[categ_id]
            else:
                key = categ_id

            if with_score:
                det[key] = _bboxes, _scores
            else:
                det[key] = _bboxes
        pr.append(det)
    return pr


if __name__ == "__main__":
    categs = np.array([0, 1, 1, 2, 2, 2])
    bboxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [1, 1, 21, 21],
            [2, 2, 12, 12],
            [2, 2, 22, 22],
            [2, 2, 32, 32],
        ]
    )
    print(dataset_to_dict(bboxes, categs))
    # =>
    # [
    #     {0: array([[0, 0, 10, 10]])},
    #     {1: array([[1, 1, 11, 11]])},
    #     {1: array([[1, 1, 21, 21]])},
    #     {2: array([[2, 2, 12, 12]])},
    #     {2: array([[2, 2, 22, 22]])},
    #     {2: array([[2, 2, 32, 32]])},
    # ]

    decode_vals = {
        "batch_idx": np.array([0, 0, 1, 1, 1, 1]),
        "categs": np.array([2, 1, 3, 4, 1, 3]),
        "scores": np.array([0.2, 0.1, 0.3, 0.4, 0.1, 0.33]),
        "bboxes": np.array(
            [
                [2, 2, 20, 20],
                [1, 1, 10, 10],
                [3, 3, 30, 30],
                [4, 4, 40, 40],
                [1, 1, 10, 10],
                [3, 3, 300, 300],
            ]
        ),
    }
    print(decode_vals_to_dict(decode_vals))
    # =>
    # [
    #     {
    #         1: array([[1, 1, 10, 10]]),
    #         2: array([[2, 2, 20, 20]])
    #     },
    #     {
    #         1: array([[1, 1, 10, 10]]),
    #         3: array([[3, 3, 300, 300], [3, 3, 30, 30]]),
    #         4: array([[4, 4, 40, 40]]),
    #     },
    # ]
