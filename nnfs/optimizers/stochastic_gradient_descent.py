import numpy as np


class SGD:
    def __init__(self, learning_rate=1.0, decay_rate=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0

    def pre_update_parameters(self):
        if self.decay_rate:
            self.current_learning_rate = self.learning_rate * \
                (1.0 / (1.0 + self.decay_rate * self.iterations))

    def update_parameters(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    def post_update_parameters(self):
        self.iterations += 1