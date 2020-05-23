from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
import torch
from torch import Tensor


class Preprocessing:
    def __init__(self):
        self.name = None

    def transform(self, input: np.ndarray) -> Tensor:
        raise NotImplementedError


class TimeStampPreprocessing(Preprocessing):
    def __init__(self):
        pass


class Word2VecPreprocessing(Preprocessing):
    def __init__(self):
        pass


class CategoricalPrerprocessing(Preprocessing):
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(handle_unknown="error")  # unknown is all zero.
    
    def fit(self, X: List[List[str]]) -> None:
        """X: array-like, shape[n_samples, n_features]
        """
        self.one_hot_encoder.fit(X)

    def transform(self, X: List[List[str]]) -> Tensor:
        """X: array-like, shape[n_samples, n_features]
        """
        return torch.tensor(self.one_hot_encoder.transform(X), dtype=torch.float32)


class FrequencyPreprocessing(Preprocessing):
    def __init__(self):
        self.standard_scalar = StandardScaler()
    
    def fit(self, X: np.ndarray):
        self.standard_scalar.fit(X)
        
    def transform(self, X: np.ndarray) -> Tensor:
        if not isinstance(X, np.ndarray):
            raise TypeError("The input X should be nd.nparray.")
        if not (len(X.shape) == 2 and X.shape[0] == 1):
            raise ValueError(f"The shape of X should be (1, -1), but got {X.shape}")
        return torch.tensor(self.standard_scalar.transform(X), dtype=torch.float32)
