import numpy as np

class Conv2DLayer:
    def __init__(self, weights, biases, stride=(1, 1), padding='same'):
        
        self.weights = weights
        self.biases = biases
        self.stride_height, self.stride_width = stride
        self.padding = padding
        self.kernel_height, self.kernel_width, self.in_channels, self.out_channels = weights.shape
        
        if self.in_channels != biases.shape[0] / (self.out_channels / self.in_channels) and self.weights.shape[2] != self.biases.shape[0] / (self.weights.shape[3]/self.weights.shape[2]) : # Check for bias shape mistakes
             pass

    def forward(self, input_data):
        input_height, input_width, input_channel = input_data.shape
        assert input_channel == self.in_channels, "Input channels mismatch"

        # Same indicating there is padding
        if self.padding == 'same':
            kernel_padding_height = (self.kernel_height - 1) // 2
            kernel_padding_width = (self.kernel_width - 1) // 2

            if self.stride_height > 1 or self.stride_width > 1:
                # Calculate output size for 'same' padding with stride
                output_height = int(np.ceil(float(input_height) / float(self.stride_height)))
                output_width = int(np.ceil(float(input_width) / float(self.stride_width)))
                
                # Calculate required padding
                total_padding_height = max((output_height - 1) * self.stride_height + self.kernel_height - input_height, 0)
                total_padding_width = max((output_width - 1) * self.stride_width + self.kernel_width - input_width, 0)

                padding_top = total_padding_height // 2
                padding_bottom = total_padding_height - padding_top
                padding_left = total_padding_width // 2
                padding_right = total_padding_width - padding_left
            else: # stride = 1
                output_height = input_height
                output_width = input_width
                padding_top = kernel_padding_height
                padding_bottom = kernel_padding_height
                padding_left = kernel_padding_width
                padding_right = kernel_padding_width

            padded_input = np.pad(input_data, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), mode='constant')

        # No Padding
        elif self.padding == 'valid':
            padded_input = input_data
            output_height = (input_height - self.kernel_height) // self.stride_height + 1
            output_width = (input_width - self.kernel_width) // self.stride_width + 1
        else:
            raise ValueError("Unsupported padding type")
            
        output_data = np.zeros((output_height, output_width, self.out_channels))

        for output_channel in range(self.out_channels): 
            for row_idx in enumerate(range(0, output_height)): 
                for col_idx in enumerate(range(0, output_width)): 
                    row_start = row_idx * self.stride_height
                    row_end = row_start + self.kernel_height
                    col_start = col_idx * self.stride_width
                    col_end = col_start + self.kernel_width
                    
                    
                    # Perform convolution operation
                    conv_sum = np.sum(padded_input[row_start:row_end, col_start:col_end, :] * self.weights[:, :, :, output_channel])
                    output_data[row_idx, col_idx, output_channel] = conv_sum + self.biases[output_channel]
                
        return output_data

class ReLULayer:
    def forward(self, input_data):
        return np.maximum(0, input_data)

class MaxPooling2DLayer:
    def __init__(self, pool_size=(2, 2), stride=None):
        self.pool_height, self.pool_width = pool_size
        self.stride_height, self.stride_width = stride if stride is not None else pool_size

    def forward(self, input_data):
        # input_data shape: (height, width, channels)
        input_height, input_width, num_channels = input_data.shape
        
        output_height = (input_height - self.pool_height) // self.stride_height + 1
        output_width = (input_width - self.pool_width) // self.stride_width + 1
        
        output_data = np.zeros((output_height, output_width, num_channels))
        
        for channel in range(num_channels):
            for height_idx, output_row in enumerate(range(0, output_height)):
                for width_idx, output_col in enumerate(range(0, output_width)):
                    height_start = height_idx * self.stride_height
                    height_end = height_start + self.pool_height
                    width_start = width_idx * self.stride_width
                    width_end = width_start + self.pool_width
                    
                    window = input_data[height_start:height_end, width_start:width_end, channel]
                    output_data[output_row, output_col, channel] = np.max(window)
        return output_data
    
class AveragePooling2DLayer:
    def __init__(self, pool_size=(2, 2), stride=None):
        self.pool_height, self.pool_width = pool_size
        self.stride_height, self.stride_width = stride if stride is not None else pool_size

    def forward(self, input_data):
        input_height, input_width, num_channels = input_data.shape
        
        output_height = (input_height - self.pool_height) // self.stride_height + 1
        output_width = (input_width - self.pool_width) // self.stride_width + 1
        
        output_data = np.zeros((output_height, output_width, num_channels))
        
        for channel in range(num_channels):
            for height_idx, output_row in enumerate(range(0, output_height)):
                for width_idx, output_col in enumerate(range(0, output_width)):
                    height_start = height_idx * self.stride_height
                    height_end = height_start + self.pool_height
                    width_start = width_idx * self.stride_width
                    width_end = width_start + self.pool_width
                    
                    window = input_data[height_start:height_end, width_start:width_end, channel]
                    output_data[output_row, output_col, channel] = np.mean(window)
        return output_data


class FlattenLayer:
    def forward(self, input_data):
        return input_data.flatten() 

class GlobalAveragePooling2DLayer:
    def forward(self, input_data):
        return np.mean(input_data, axis=(0, 1))

class DenseLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, input_data):
        return np.dot(input_data, self.weights) + self.biases
    
# Return the result in SoftMax layer
class SoftmaxLayer:
    def forward(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data)) 
        return exp_values / np.sum(exp_values)
    
class TanhLayer:
    def forward(self, input_data):
        return np.tanh(input_data)
    
class SigmoidLayer:
    def forward(self, input_data):
        input_data = np.clip(input_data, -500, 500)
        return 1 / (1 + np.exp(-input_data))