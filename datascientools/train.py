import pandas as pd

from dataset.data_loader import DataLoader, YoutubeTrendingDataset
from preprocessing.preprocessing import (
    TimeStampPreprocessing,
    Word2VecPreprocessing,
    CategoricalPrerprocessing,
    FrequencyPreprocessing
)
from preprocessing.preprocessor import Preprocessor


def train():
    train_dataset = YoutubeTrendingDataset("./data/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    preprocess_dict = {
        "publish_time": [TimeStampPreprocessing],
        "likes": [FrequencyPreprocessing],
        "category_id": [CategoricalPrerprocessing]
    }
    preprocessor = Preprocessor(preprocess_dict)

    for batch_ix, data in enumerate(train_dataloader):
        train_x, train_y = preprocessor.run(data)


if __name__ == "__main__":
    train()
