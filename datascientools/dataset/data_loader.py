from pathlib import Path
from typing import Dict

import pandas as pd
from pandas import Series

import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.dataloader


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int):
        super(DataLoader, self).__init__(dataset, batch_size)


class YoutubeTrendingDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, index: int = 0) -> Dict:
        return self.df.loc[index].to_dict()

    def __len__(self) -> int:
        return len(self.df)
