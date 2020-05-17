import unittest

from datascientools.dataset.data_loader import YoutubeTrendingDataset, DataLoader


class TestDataLoader(unittest.TestCase):
    def test_load_youtube_trending_dataset(self):
        youtube_trending_dataset = YoutubeTrendingDataset("./data/train.csv")
        first_row = youtube_trending_dataset[0]
        self.assertEqual(type(first_row).__name__, dict.__name__)


if __name__ == '__main__':
    unittest.main()
