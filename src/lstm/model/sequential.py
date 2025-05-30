class ModelScratch:
    """Model Sequential dari scratch."""
    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        """Melakukan prediksi (forward pass) untuk seluruh model."""
        for layer in self.layers:
            x = layer.forward(x)
        return x