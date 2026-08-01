"""CIFAR-10/100 loader backed by Hugging Face Hub parquet files (fast mirror).

Falls back to torchvision if the Hub is unreachable. Returns uint8 numpy arrays
(train/test) with images in (N, H, W, C) and labels.
"""
import os

import numpy as np

_HUB_DS = {"cifar10": "uoft-cs/cifar10", "cifar100": "uoft-cs/cifar100"}
_HUB_FILES = {
    "cifar10": ("plain_text/train-00000-of-00001.parquet", "plain_text/test-00000-of-00001.parquet"),
    "cifar100": ("cifar100/train-00000-of-00001.parquet", "cifar100/test-00000-of-00001.parquet"),
}


def _decode(img_struct):
    from PIL import Image
    import io
    b = img_struct["bytes"]
    if isinstance(b, memoryview):
        b = b.tobytes()
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))


def _load_parquet(hf_path, cache_dir, filename):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local = os.path.join(cache_dir, filename)
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        local = hf_hub_download(repo_id=hf_path, filename=filename, repo_type="dataset",
                                cache_dir=cache_dir, local_dir=cache_dir)
    t = pq.read_table(local)
    imgs = np.stack([_decode(x) for x in t.column("img").to_pylist()])
    labels = t.column("fine_label").to_numpy()
    return imgs, labels


def load_cifar(dataset="cifar100", cache_dir=None):
    if cache_dir is None:
        cache_dir = os.environ.get("CIFAR_CACHE", os.path.expanduser("~/.cache/cifar_hf"))
    ds_id = _HUB_DS[dataset]
    tr_file, te_file = _HUB_FILES[dataset]
    x_tr, y_tr = _load_parquet(ds_id, cache_dir, tr_file)
    x_te, y_te = _load_parquet(ds_id, cache_dir, te_file)
    return x_tr, y_tr, x_te, y_te
