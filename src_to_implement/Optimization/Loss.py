import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.output = None
        self.temp_x=None

    def forward(self, prediction_tensor, label_tensor):
        self.temp_x= prediction_tensor.clip(min=1e-8, max=None)
        self.output = (np.where(label_tensor == 1, -np.log(self.temp_x), 0)).sum(axis=1)
        return self.output

    def backward(self, label_tensor):
        error_tensor = np.where(label_tensor == 1, -1 / self.temp_x, 0)
        return error_tensor
