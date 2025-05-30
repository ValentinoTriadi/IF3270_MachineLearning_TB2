import time
from cnn.layers import Conv2DLayer, ReLULayer, MaxPooling2DLayer, FlattenLayer, GlobalAveragePooling2DLayer, AveragePooling2DLayer, DenseLayer, SoftmaxLayer, TanhLayer, SigmoidLayer

import numpy as np
from tensorflow import keras

class ScratchCNNModel:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer_instance):
        self.layers.append(layer_instance)

    def load_keras_model(self, keras_model_instance):
        print("Load Keras Model ")
        for keras_layer in keras_model_instance.layers:
            
            # Handle Convolution layer
            if isinstance(keras_layer, keras.layers.Conv2D):
                
                weights, biases = keras_layer.get_weights()
                
                # Stride bisa tuple atau int. Pastikan itu tuple (s_h, s_w)
                stride = keras_layer.strides 
                if isinstance(stride, int): stride = (stride, stride)

                scratch_conv = Conv2DLayer(weights, biases, stride=stride, padding=keras_layer.padding)
                self.add_layer(scratch_conv)
                print(f"Adding Conv2DLayer (kernel: {weights.shape}, stride: {stride}, padding: {keras_layer.padding})")
                
                if keras_layer.activation.__name__ == 'relu':
                    self.add_layer(ReLULayer())
                    print(f"Adding ReLULayer")
                elif keras_layer.activation.__name__ != 'linear': 
                    print(f"WARN: Aktivasi {keras_layer.activation.__name__} pada Conv2D belum didukung secara eksplisit setelah Conv.")

            # Handle Max pooling 
            elif isinstance(keras_layer, keras.layers.MaxPooling2D):
                stride = keras_layer.strides
                if isinstance(stride, int): stride = (stride, stride)
                pool_size = keras_layer.pool_size

                scratch_pool = MaxPooling2DLayer(pool_size=pool_size, stride=stride)
                self.add_layer(scratch_pool)
                print(f"Adding MaxPooling2DLayer (pool_size: {pool_size}, stride: {stride})")

            elif isinstance(keras_layer, keras.layers.AveragePooling2D):
                stride = keras_layer.strides
                if isinstance(stride, int): stride = (stride, stride)
                pool_size = keras_layer.pool_size
                
                scratch_pool = AveragePooling2DLayer(pool_size=pool_size, stride=stride)
                self.add_layer(scratch_pool)
                print(f"Adding AveragePooling2DLayer (pool_size: {pool_size}, stride: {stride})")

            # Handle Average Pooling
            elif isinstance(keras_layer, keras.layers.GlobalAveragePooling2D):
                self.add_layer(GlobalAveragePooling2DLayer())
                print(f"Adding GlobalAveragePooling2DLayer")

            # Handle Flatten Layer
            elif isinstance(keras_layer, keras.layers.Flatten):
                self.add_layer(FlattenLayer())
                print(f"Adding FlattenLayer")

            # Handle Dense/FFNN Layer
            elif isinstance(keras_layer, keras.layers.Dense):
                weights, biases = keras_layer.get_weights()
                scratch_dense = DenseLayer(weights, biases)
                self.add_layer(scratch_dense)
                print(f"Adding Dense Layer(weight: {weights.shape})")

                if keras_layer.activation.__name__ == 'relu':
                    self.add_layer(ReLULayer())
                    print(f"Adding ReLu")
                elif keras_layer.activation.__name__ == 'softmax':
                    self.add_layer(SoftmaxLayer())
                    print(f"Adding SoftMax")

                elif keras_layer.activation.__name__ == 'tanh':
                    self.add_layer(TanhLayer())
                    print(f"Adding Tanh")

                elif keras_layer.activation.__name__ == 'sigmoid':
                    self.add_layer(SigmoidLayer())
                    print("Adding sigmoid")

                elif keras_layer.activation.__name__ != 'linear':
                     print(f"WARN: Activation for {keras_layer.activation.__name__} aren't supported.")

            else:
                print(f"WARN: Layer of type {keras_layer.__class__.__name__} aren't supported.")

    def predict_single(self, input_sample, verbose=False):
        current_output = input_sample
        if verbose:
            print(f"Forward propagating single sample through {len(self.layers)} layers")
            print(f"Initial input shape: {current_output.shape}")
        
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"Applying layer {i+1:2d}/{len(self.layers)}: {layer.__class__.__name__:<25}", end="")
            
            prev_shape = current_output.shape
            current_output = layer.forward(current_output)
        
        if verbose:
            print(f"Single sample prediction complete. Final output shape: {current_output.shape}")
        return current_output

    def predict_batch(self, input_batch, model_name_tag="ScratchModel"):
        predictions = []
        total_samples = len(input_batch)
        
        if total_samples == 0:
            print("Input batch is empty. No predictions to make.")
            return np.array([])

        print(f"\nStarting Scratch Prediction for '{model_name_tag}' on {total_samples} samples:")
        
        start_time = time.time()
        for i, sample in enumerate(input_batch):
            # Update progress bar
            progress = (i + 1) / total_samples
            
            print(f"Processing: {i+1}/{total_samples} ({progress*100:.1f}%)", end='\r')
            
            predictions.append(self.predict_single(sample, verbose=False))
        
        end_time = time.time()
        total_time = end_time - start_time

        print()
        print(f"Batch prediction complete for '{model_name_tag}'.")
        print(f"Total time: {total_time:.2f} seconds ({total_time/total_samples:.3f} sec/sample).")
        print(f"Output array shape: {np.array(predictions).shape}")
        print()
        
        return np.array(predictions)