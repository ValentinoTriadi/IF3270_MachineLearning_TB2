import numpy as np

from .lstm import LSTMLayerScratch

class BidirectionalScratch:
    def __init__(self, units, all_weights):
        # all_weights adalah list [W_f, U_f, b_f, W_b, U_b, b_b]
        self.forward_lstm = LSTMLayerScratch(units, all_weights[:3])
        self.backward_lstm = LSTMLayerScratch(units, all_weights[3:])

    def forward(self, x):
        """
        Melakukan forward propagation.
        Input x: array float (batch_size, sequence_length, input_dim).
        Output: array float (batch_size, 2 * units).
        """
        # Arah maju
        h_forward = self.forward_lstm.forward(x)
        # Arah mundur (input dibalik)
        h_backward = self.backward_lstm.forward(x[:, ::-1, :])
        # Gabungkan hasil
        return np.concatenate((h_forward, h_backward), axis=-1)