from torch.utils.data import Dataset
import lmdb
import pyarrow as pa
from PIL import Image


class ImageLMDBDataset(Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b"__len__"))
            self.keys = pa.deserialize(txn.get(b"__keys__"))

        self.transform = transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        img_np = unpacked[0]
        img = Image.fromarray(img_np)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length
