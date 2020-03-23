import os
from argparse import Namespace

import numpy as np
import torchvision

import fire
import lmdb
import pyarrow as pa
from datasets import ImagePathsDataset
import torch


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def get_dataloaders(args, collate_fn=None):
    if args.dataset_type == "folder":
        train = torchvision.datasets.ImageFolder(args.dataset_path)
    if args.dataset_type == "paths":
        train = ImagePathsDataset(args.dataset_path)

    loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader


def main(lmdb_path, dataset_type, dataset_path, write_frequency=5000, num_workers=16):
    args = Namespace(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        batch_size=1,
        num_workers=num_workers,
    )

    loader = get_dataloaders(args, collate_fn=lambda x: x)

    os.makedirs(lmdb_path, exist_ok=True)
    db = lmdb.open(
        lmdb_path,
        subdir=True,
        map_size=1099511627776 * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)

    for idx, data in enumerate(loader):
        # print(type(data), data)
        image, label = data[0]
        image = np.array(image)
        txn.put(u"{}".format(idx).encode("ascii"), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u"{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_pyarrow(keys))
        txn.put(b"__len__", dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    fire.Fire(main)
