class Sgd:
    def __init__(self, learning_rate):
        assert(type(learning_rate) == float or type(learning_rate) == int)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """ Stochastic gradient descent """
        updated_weights = weight_tensor-(self.learning_rate * gradient_tensor)
        return updated_weights
