import unittest

import numpy as np
import torch

from datascientools.preprocessing.preprocessing import (
    CategoricalPrerprocessing,
    FrequencyPreprocessing,
)


# class TestTimeStampPreprocessing(unittest.TestCase):
#     def setUp(self):
#         self.timestamp_preprocessing = TimeStampPreprocessing()

#     def test_timestamp_to_tensor(self):
#         self.assertEqual(self.timestamp_preprocessing.run(
#             "2020-02-18"), torch.LongTensor(20200218))
#         self.assertEqual(self.timestamp_preprocessing.run(
#             "2018-03-12T09:17:28.000Z", torch.LongTensor(20180312)))

  
class TestFreqPreprocessing(unittest.TestCase):
    def test_freq_preprocessing(self):
        freq_preprocessing = FrequencyPreprocessing()
        freq_preprocessing.fit(np.array([[0., 1., 0., 3.], [0., 1., 1., 0.]]))
        self.assertTrue(torch.allclose(
            freq_preprocessing.transform(np.array([[1., 1., 1., 1.]])),
            torch.tensor([[1., 0, 1, -0.333333]], dtype=torch.float32))
        )

class TestCategoricalPreprocessing(unittest.TestCase):
    def test_categorical_preprocessing(self):
        categorical_preprocessing = CategoricalPrerprocessing()
        categorical_preprocessing.fit([["a"], ["b"], ["c"], ["b"], ["d"]])
        print(categorical_preprocessing.one_hot_encoder.categories_)
        print(categorical_preprocessing.transform([["a"], ["d"], ["c"], ["d"]]))
        self.assertTrue(torch.allclose(
            categorical_preprocessing.transform([["a"], ["d"], ["c"], ["d"]]),
            torch.Tensor([
                [1., 0., 0., 0.], 
                [0., 0., 0., 1.], 
                [0., 0., 1., 0.], 
                [0., 0., 0., 1.], 
            ], dtype=torch.float32)
        ))

if __name__ == "__main__":
    unittest.main()
