from typing import List, Tuple

import torch
from torch import Tensor


class Preprocessing:
    def __init__(self):
        self.name = None

    def run(self, *input) -> Tensor:
        #raise NotImplementedError
        print(self.__str__())
        return torch.zeros([1])


class TimeStampPreprocessing(Preprocessing):
    def __init__(self):
        pass


class Word2VecPreprocessing(Preprocessing):
    def __init__(self):
        pass


class CategoricalPrerprocessing(Preprocessing):
    def __init__(self):
        pass


class FrequencyPreprocessing(Preprocessing):
    def __init__(self):
        pass
