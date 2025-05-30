import numpy as np

from .fungsi_aktivasi import relu, softmax

class DenseScratch:
    """Layer Dense (Fully Connected) dari scratch."""
    def __init__(self, weights, activation_name=None):
        # weights adalah list [W, b]
        self.W, self.b = weights
        self.activation_name = activation_name

    def forward(self, x):
        """
        Melakukan forward propagation.
        Input x: array float (batch_size, input_features).
        Output: array float (batch_size, output_features).
        """
        z = np.dot(x, self.W) + self.b
        if self.activation_name == 'relu':
            return relu(z)
        elif self.activation_name == 'softmax':
            return softmax(z)
        else:
            return z