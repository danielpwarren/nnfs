import numpy as np


class Step:
    def forward(self, inputs):
        self.output = np.where(inputs > 0, 1, 0)
