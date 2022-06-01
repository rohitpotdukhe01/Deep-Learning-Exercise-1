class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, input_tensor):

        for input in input_tensor:
            if input > 0:
                self.output.append(input)
            else:
                self.output.append(0)
        return self.output

    def backward(self, error_tensor):

        pass
