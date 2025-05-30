class EmbeddingScratch:
    def __init__(self, weights):
        self.embedding_matrix = weights[0]
        # self.embedding_matrix[0] = np.zeros_like(self.embedding_matrix[0])

    def forward(self, x):
        """
        Melakukan forward propagation.
        Input x: array integer (batch_size, sequence_length).
        Output: array float (batch_size, sequence_length, embedding_dim).
        """
        return self.embedding_matrix[x]