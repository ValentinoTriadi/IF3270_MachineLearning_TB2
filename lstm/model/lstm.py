import numpy as np
from .fungsi_aktivasi import sigmoid, tanh

class LSTMLayerScratch:
    def __init__(self, units, weights):
        # weights adalah list [W, U, b]
        W, U, b = weights
        self.units = units

        # Membagi matriks bobot menjadi 4 gate (input, forget, cell, output)
        self.W_i = W[:, :units]
        self.W_f = W[:, units:units*2]
        self.W_c = W[:, units*2:units*3]
        self.W_o = W[:, units*3:]

        self.U_i = U[:, :units]
        self.U_f = U[:, units:units*2]
        self.U_c = U[:, units*2:units*3]
        self.U_o = U[:, units*3:]

        self.b_i = b[:units]
        self.b_f = b[units:units*2]
        self.b_c = b[units*2:units*3]
        self.b_o = b[units*3:]

    def forward(self, x):
        """
        Melakukan forward propagation.
        Input x: array float (batch_size, sequence_length, input_dim).
        Output: array float (batch_size, units) - Hanya state terakhir.
        """
        batch_size, seq_len, _ = x.shape
        h_t = np.zeros((batch_size, self.units))
        c_t = np.zeros((batch_size, self.units))

        # Iterasi melalui setiap timestep
        for t in range(seq_len):
            x_t = x[:, t, :]

            # Hitung gate
            i_t = sigmoid(np.dot(x_t, self.W_i) + np.dot(h_t, self.U_i) + self.b_i)
            f_t = sigmoid(np.dot(x_t, self.W_f) + np.dot(h_t, self.U_f) + self.b_f)
            o_t = sigmoid(np.dot(x_t, self.W_o) + np.dot(h_t, self.U_o) + self.b_o)
            c_tilde_t = tanh(np.dot(x_t, self.W_c) + np.dot(h_t, self.U_c) + self.b_c)

            # Hitung cell state dan hidden state baru
            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * tanh(c_t)
            
        return h_t
