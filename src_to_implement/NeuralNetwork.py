import copy as cp


class NeuralNetwork():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.next_input = None
        self.previous_input = None
        self.output = None

    def forward(self):
        self.next_input, self.previous_input = self.data_layer.next()
        input_copy = self.next_input.copy()
        for layer in self.layers:
            input_copy = layer.forward(input_copy)
        self.output = self.loss_layer.forward(input_copy, self.previous_input)
        return self.output

    def backward(self):
        loss_gradiant = self.loss_layer.backward(self.previous_input)

        for layer in self.layers[::-1]:
            loss_gradiant = layer.backward(loss_gradiant)
        # return loss_gradiant

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = cp.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        inp = input_tensor
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp
