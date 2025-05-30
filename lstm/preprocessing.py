import keras


class LSTMPreprocess:
    def __init__(self):
        self.embedding_layer = keras.Sequential()
        self.vectorize_layer = None

    def tokenization(self, input, vocab_size=10000, max_len=100, **kwargs):
        self.vectorize_layer = keras.layers.TextVectorization(
            max_tokens=vocab_size,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            ngrams=None,
            output_mode="int",
            output_sequence_length=max_len,
            pad_to_max_tokens=False,
            vocabulary=None,
            idf_weights=None,
            sparse=False,
            ragged=False,
            encoding="utf-8",
            name=None,
            **kwargs
        )
        self.vectorize_layer.adapt(input)
        return self.vectorize_layer(input)

    def embedding(self, token, output_length=10, **kwargs):
        input_length = len(self.vectorize_layer.get_vocabulary())
        self.embedding_layer.add(
            keras.layers.Embedding(
                input_length,
                output_length,
                embeddings_initializer="uniform",
                embeddings_regularizer=None,
                embeddings_constraint=None,
                mask_zero=False,
                weights=None,
                lora_rank=None,
                lora_alpha=None,
                **kwargs
            )
        )
        self.embedding_layer.compile("rmsprop", "mse")
        output_array = self.embedding_layer.predict(token)
        return output_array


if __name__ == "__main__":
    cls = LSTMPreprocess()
    input = input("MASUKIN WOI: ")
    token = cls.tokenization(input)
    print(token)
    embed = cls.embedding(token)
    print(embed)
