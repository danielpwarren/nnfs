import numpy as np


# Abstract loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, np.array(y))
        data_loss = np.mean(sample_losses)
        return data_loss
