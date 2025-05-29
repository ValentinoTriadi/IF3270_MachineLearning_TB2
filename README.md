# RNN-based Text Classification on NusaX-Sentiment Dataset

This project implements text classification on the NusaX-Sentiment Indonesian dataset using RNN-based models. It includes data preprocessing, model training, hyperparameter variation analysis, and a custom **Simple RNN** forward propagation implementation from scratch.

## Features

- Text preprocessing with `TextVectorization` and `Embedding` layers
- RNN-based models with configurable layers, units, and directionality
- Comprehensive evaluation using accuracy and macro F1-score
- Visualization of training history
- Hyperparameter variation experiments
- **Simple RNN** forward propagation implementation from scratch
- Customizable activation functions for RNN cells (separate for hidden state and output)

## Dataset

The project uses the NusaX-Sentiment Indonesian dataset in CSV format, which contains:
- Text data in Indonesian language
- Sentiment labels (positive, negative, neutral)

The dataset is automatically downloaded from the [NusaX GitHub repository](https://github.com/IndoNLP/nusax/tree/main/datasets/sentiment/indonesian) when running the script.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn
- requests

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Run the main script to execute all experiments:

```bash
python rnn_text_classification.py
```

This will:
1. Download the NusaX-Sentiment Indonesian dataset (CSV files)
2. Preprocess the text data
3. Run experiments with different hyperparameter configurations:
   - Varying number of RNN layers (1, 2, 3)
   - Varying number of RNN units per layer (32, 64, 128)
   - Comparing bidirectional vs unidirectional RNNs
4. Save trained models in the `models/` directory
5. Save training history plots in the `figures/` directory
6. Save experiment results in the `results/` directory

### Simple RNN Forward Propagation from Scratch

After running the experiments, you can run the custom Simple RNN forward propagation implementation:

```bash
python rnn_forward_propagation.py --activation tanh
```

This will:
1. Load the best-performing model based on the experiment results
2. Extract the model weights and architecture
3. Implement **Simple RNN** forward propagation from scratch using NumPy
   - The implementation uses basic/vanilla RNN cells (not LSTM or GRU)
   - The code is modular with separate functions for each layer type
   - You can specify different activation functions using the `--activation` parameter
4. Compare the predictions from the Keras model and the scratch implementation
5. Generate comparison visualizations in the `figures/` directory

The script provides detailed step-by-step output during the forward propagation process.

#### Customizable Activation Functions

The RNN implementation now supports customizable activation functions for both hidden state updates (h_t) and outputs (y_t):

```python
# In rnn_forward_propagation.py, modify these values:
hidden_activation = 'tanh'   # Activation for hidden state updates (h_t)
output_activation = 'tanh'   # Activation for outputs (y_t)
```

You can use different combinations:
- Hidden state with tanh, output with sigmoid
- Hidden state with relu, output with tanh
- Any other combination of the supported activation functions

The activation functions are implemented in a separate module (`activation_functions.py`) which can be easily extended with additional functions.

### Visualizing Results

After running the experiments, you can visualize the results using:

```bash
python visualize_results.py
```

This will generate additional plots comparing the performance of different model configurations in the `figures/` directory.

## Directory Structure

```
.
├── data/                       # Dataset files (CSV)
├── figures/                    # Training history plots and visualizations
├── models/                     # Saved model files
├── results/                    # Experiment results
├── activation_functions.py     # Customizable activation functions
├── rnn_text_classification.py  # Main script
├── rnn_forward_propagation.py  # Custom Simple RNN implementation
├── visualize_results.py        # Results visualization script
├── requirements.txt            # Required packages
└── README.md                   # This file
```

## Experiments

The script runs three main experiments:

1. **Impact of Number of RNN Layers**: Comparing models with 1, 2, and 3 RNN layers
2. **Impact of Number of RNN Cells per Layer**: Comparing models with 32, 64, and 128 units per layer
3. **Impact of RNN Layer Type by Direction**: Comparing bidirectional vs unidirectional RNNs

Results are evaluated using the macro F1-score metric and saved to `results/experiment_results.json`.

## Custom Simple RNN Implementation

The `rnn_forward_propagation.py` script implements forward propagation using a basic/vanilla RNN from scratch using NumPy for the best-performing model. It includes:

1. **Weight Extraction**: Extracting weights from the trained Keras model and adapting them for Simple RNN
2. **Modular Implementation**: Separate functions for each layer type (Embedding, Simple RNN, Bidirectional, Dense)
3. **Step-by-Step Execution**: Detailed output of each step in the forward propagation process
4. **Customizable Activation Functions**: Support for different activation functions for hidden state updates (h_t) and outputs (y_t)
5. **Performance Comparison**: Comparing predictions from the Keras model and the scratch implementation
6. **Visualization**: Generating plots to visualize the comparison

### Activation Functions

The implementation supports the following activation functions for both hidden state (h_t) and output (y_t):

- **tanh**: Hyperbolic tangent function (default for RNN)
- **sigmoid**: Sigmoid function (0 to 1 range)
- **relu**: Rectified Linear Unit (no upper bound)

In a traditional RNN, the same activation function is used for both the hidden state update and the output. However, in this implementation, you can use different activation functions for each, allowing for more flexible RNN architectures.

For example:
- Using tanh for hidden state updates and sigmoid for outputs
- Using relu for hidden state updates and tanh for outputs

This can lead to different model behaviors and potentially better performance on certain tasks.

## Output

The program generates:
- Model files (.h5) in the `models/` directory
- Training history plots for each experiment in the `figures/` directory
- Summary visualizations comparing all experiments
- Detailed results in JSON format in the `results/` directory
- Comparison visualizations between Keras and Simple RNN from scratch implementations
