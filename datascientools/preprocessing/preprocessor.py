from typing import Dict, List, Tuple

import torch
from torch import Tensor


class Preprocessor:
    def __init__(self, preprocess_dict: Dict):
        self.preprocess_dict = preprocess_dict

    def run(self, data) -> Tuple[Tensor, Tensor]:
        for key, preprocess_list in self.preprocess_dict.items():
            for preprocess in preprocess_list:
                preprocess.run(data[key])
        return torch.zeros([1]), torch.zeros([1])
