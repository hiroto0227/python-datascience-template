import unittest

import torch

from datascientools.preprocessing.preprocessing import (
    TimeStampPreprocessing
)


class TestTimeStampPreprocessing(unittest.TestCase):
    def setUp(self):
        self.timestamp_preprocessing = TimeStampPreprocessing()

    def test_timestamp_to_tensor(self):
        self.assertEqual(self.timestamp_preprocessing.run(
            "2020-02-18"), torch.LongTensor(20200218))
        self.assertEqual(self.timestamp_preprocessing.run(
            "2018-03-12T09:17:28.000Z", torch.LongTensor(20180312)))


if __name__ == "__main__":
    unittest.main()
