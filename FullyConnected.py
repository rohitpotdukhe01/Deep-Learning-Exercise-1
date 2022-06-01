from .Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        self.__optimizer = val

    def gradient_weights(self):
        return self.__gradient_weights

    def gradient_weights(self, val):
        self.__gradient_weights = val

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(0, 1)
        self.gradient_weights = None
        self.optimizers = None

    def forward(self, input_tensor):
        bias = [1] * input_tensor.shape[0]
        bias_array = np.array(bias).reshape(input_tensor.shape[0], 1)
        d = np.concatenate(input_tensor, bias_array)
        output = np.dot(d, self.weights)
        return output

    def backward(self, error_tensor):
        self.output = np.dot(error_tensor, self.weights[:-1].T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.output
