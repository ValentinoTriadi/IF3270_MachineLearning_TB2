import numpy as np

def tanh(x):
    """
    Hyperbolic tangent activation function.
    
    Args:
        x: Input tensor
    
    Returns:
        tanh(x)
    """
    return np.tanh(x)

def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x: Input tensor
    
    Returns:
        sigmoid(x)
    """
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow

def relu(x):
    """
    Rectified Linear Unit activation function.
    
    Args:
        x: Input tensor
    
    Returns:
        max(0, x)
    """
    return np.maximum(0, x)

def softmax(x, axis=-1):
    """
    Softmax activation function.
    
    Args:
        x: Input tensor
        axis: Axis along which to apply softmax
    
    Returns:
        softmax(x)
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def get_activation_function(name):
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function ('tanh', 'sigmoid', 'relu', 'softmax')
    
    Returns:
        Activation function
    """
    activation_functions = {
        'tanh': tanh,
        'sigmoid': sigmoid,
        'relu': relu,
        'softmax': softmax
    }
    
    if name.lower() not in activation_functions:
        raise ValueError(f"Activation function '{name}' not found. Available: {list(activation_functions.keys())}")
    
    return activation_functions[name.lower()] 