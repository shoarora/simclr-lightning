from torch.utils.data import Dataset
from PIL import Image


class ImagePathsDataset(Dataset):
    def __init__(self, paths_file, transform=None):
        self.paths_file = paths_file

        with open(paths_file) as f:
            self.paths = [line.strip() for line in f if line.strip()]

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = pil_loader(path)

        if self.transform:
            img = self.transform(img)

        return img, 0


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
