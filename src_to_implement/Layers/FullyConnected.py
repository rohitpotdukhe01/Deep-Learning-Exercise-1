from .Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0., 1., (self.input_size+1, self.output_size))
        self.gradient_weights = None
        self.optimizer = None
        self.input_tensor = None

    def forward(self, input_tensor):
        bias = [1] * input_tensor.shape[0]
        bias_array = np.array(bias).reshape(input_tensor.shape[0], 1)
        self.input_tensor = np.concatenate((input_tensor, bias_array), axis=1)
        output = np.dot(self.input_tensor, self.weights)
        assert (output.shape == (self.input_tensor.shape[0], self.output_size))
        return output

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        self.__optimizer = val

    def backward(self, error_tensor):
        self.output = np.dot(error_tensor, self.weights[:-1].T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.output
    @property
    def gradient_weights(self):
        return self.__gradient_weights

    def gradient_weights(self, val):
        self.__gradient_weights = val

