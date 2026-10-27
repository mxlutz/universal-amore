# %%
import os
import re
from glob import glob

from torch.utils.data import Dataset
import h5py
import numpy as np


def _file_sort_key(path):
    name = os.path.basename(path)
    anonymous = re.fullmatch(r"subj(\d+)_(-?\d+(?:\.\d+)?)\.h5", name)
    if anonymous:
        return int(anonymous[1]), float(anonymous[2])
    parts = name.split("_")
    return int(parts[1]), float(parts[2])


class MrfGreDataset(Dataset):
    def __init__(
        self,
        data_dir,
    ):
        self.data_dir = data_dir
        self.files = sorted(
            [f for f in glob(os.path.join(self.data_dir, "*.h5"))],
            key=_file_sort_key,
        )

    def __len__(self):
        return len(self.files)

    def _get_single(self, idx):
        file = self.files[idx]
        with h5py.File(file, "r") as f:
            b1_key = (
                "b1_map_channelwise_deg"
                if "b1_map_channelwise_deg" in f
                else "label_b1_map_channelwise_deg"
            )
            b1_channelwise_deg = f[b1_key][()]
            mask_liver = f['mask_liver'][()]

        return b1_channelwise_deg, mask_liver

    def get_stack(self, indices):
        samples = [self._get_single(idx) for idx in indices]
        if not samples:
            raise IndexError("Cannot stack an empty dataset selection.")

        b1_channelwise_deg, mask_liver = zip(*samples)
        return (
            np.stack(b1_channelwise_deg, axis=2),
            np.stack(mask_liver, axis=2),
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_stack(range(*idx.indices(len(self))))

        return self._get_single(idx)

def test():
    data_dir = "data/train"

    dataset = MrfGreDataset(data_dir)


    for i in range(len(dataset)):
        b1_channelwise_deg = dataset[i]



if __name__ == "__main__":
    test()

# %%
