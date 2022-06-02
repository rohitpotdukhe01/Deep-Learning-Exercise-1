import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.output = None
        self.temp_x = None
        self.eps = None
        self.temp_y = None
        self.prediction_array = None

    def forward(self, prediction_tensor, label_tensor):
        self.eps = np.finfo(float).eps
        self.prediction_array = np.array(prediction_tensor)
        self.temp_x = np.log(np.array(prediction_tensor) + self.eps)
        self.temp_y = np.array(label_tensor)
        self.output = np.where(self.temp_y == 1, -self.temp_x, 0).sum()
        return self.output

    def backward(self, label_tensor):
        error_tensor = np.where(label_tensor == 1, -1 / self.prediction_array, 0)
        return error_tensor

