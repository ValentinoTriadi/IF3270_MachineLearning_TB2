# CNN-RNN-LSTM-from-scratch

This project implements deep learning models using TensorFlow/Keras, including **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10** dataset, and **Recurrent Neural Network (RNN)** and **Long Short-Term Memory (LSTM)** for text classification on the **NusaX-Sentiment** Indonesian dataset. It also includes a custom **CNN**, **RNN**, and **LSTM** forward propagation implemented from scratch using NumPy.

---

## Features

* **CNN model** for image classification using the **CIFAR-10** dataset
* **RNN and LSTM models** for sentiment analysis on Indonesian text using the **NusaX-Sentiment** dataset
* **From-scratch forward propagation** implementations for CNN, Simple RNN, and LSTM using only NumPy (without TensorFlow/Keras during inference)
* Evaluation of models using **accuracy** and **macro F1-score**
* **Custom activation function support** for scratch implementations (e.g., ReLU, tanh, sigmoid, softmax)
* Configurable architecture options:
  * Number of layers and neurons
  * Directionality (uni/bidirectional)
  * Type of pooling (MaxPooling, AveragePooling for CNN)
  * Filter and kernel size variations for CNN
* Detailed **visualization of training process and performance**
* **Comparison between Keras and from-scratch implementations** for validation and understanding

## Datasets

### CIFAR-10 (for CNN)

The CIFAR-10 dataset is used for image classification tasks. It contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is automatically downloaded via `tensorflow.keras.datasets.cifar10`.

### NusaX-Sentiment (for RNN and LSTM)

The NusaX-Sentiment dataset contains Indonesian language text labeled as positive, negative, or neutral. It is used for text classification tasks and is downloaded from the [NusaX GitHub repository](https://github.com/IndoNLP/nusax/tree/main/datasets/sentiment/indonesian).

---

## Requirements

* Python 3.8+
* TensorFlow >= 2.8.0
* Keras
* pandas >= 1.3.0
* numpy >= 1.19.5
* matplotlib >= 3.4.0
* scikit-learn >= 1.0.0
* requests >= 2.25.0

---

## Setup Virtual Environment

1. install `virtualenv`
2. ```sh
   virtualenv venv
   ```
3. on Mac:
   ```sh
   source venv/bin/activate
   ```
   on Windows:
   ```sh
   venv/scripts/activate
   ```

---

## How To Run

### 1. Setup Virtual Environment
### 2. Install Requirements
   ```sh
   pip install -r src/requirements.txt
   ```
### Run CNN Model (Keras + From Scratch)

```bash
cd src/cnn
```

* Jalankan file Jupyter Notebook:

  * `runner.ipynb`


### Run RNN Model (Keras)

```bash
cd src/SimpleRNN
python text_classification.py
```

### Run LSTM Model (Keras)

```bash
cd src/lstm
```

* Jalankan file Jupyter Notebook:

  * `lstm.ipynb`


### Run RNN from Scratch

```bash
cd src/SimpleRNN
python forward_propagation.py
```

### Run LSTM from Scratch

```bash
cd src/lstm
```

* Jalankan file Jupyter Notebook:

  * `lstm-scratch.ipynb`

---

## Acknowledgements

| Features                                                   | PIC                          |
| ---------------------------------------------------------- | ---------------------------- |
| CNN Model & From Scratch                                   | 13522157                     |
| RNN Model & From Scratch                                   | 13522134                     |
| LSTM Model & From Scratch                                  | 13522164                     |

13522134 - Shabrina Maharani  
13522157 - Muhammad Davis Adhipramana  
13522164 - Valentino Chryslie Triadi  
