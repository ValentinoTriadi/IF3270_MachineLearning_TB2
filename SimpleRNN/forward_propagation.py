import os
import re 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, classification_report
import json
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TextVectorization
from utils.activation_functions import get_activation_function

# Mengatur seed untuk reprodusibilitas
np.random.seed(42)
tf.random.set_seed(42)

# Load vocabulary and label mapping
vocab = json.load(open('data/vocab.json', encoding='utf-8'))
token_to_index = {tok: idx for idx, tok in enumerate(vocab)}
maxlen = 100  

label_map = json.load(open('data/label_map.json'))

class RNNFromScratch:
    def __init__(self, model_path, hidden_activation='tanh', output_activation='tanh'):
        self.model_path = model_path
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        # Load Keras model
        self.keras_model = load_model(model_path)
        
        # Mengekstrak bobot dan config dari model Keras
        self.weights = {}
        self.configs = {}
        self.extract_weights()
        
        # Inisialisasi fungsi aktivasi
        self.activation_functions = {
            'tanh': lambda x: np.tanh(x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
            'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        }
        
    def extract_weights(self):
        print("\nExtracting weights from Keras model...")

        print("\nModel Summary:")
        self.keras_model.summary()
        
        # Mengekstrak bobot dari setiap layer
        for layer in self.keras_model.layers:
            if layer.name.startswith('simple_rnn'):
                print(f"\nExtracting weights for {layer.name}")
                weights = layer.get_weights()
                
                self.weights[layer.name] = {
                    'kernel': weights[0],  # W_x
                    'recurrent_kernel': weights[1],  # W_h
                    'bias': weights[2]  # b
                }
                
                self.configs[layer.name] = {
                    'units': layer.units,
                    'activation': layer.activation.__name__,
                    'use_bias': layer.use_bias,
                    'return_sequences': layer.return_sequences
                }
                
                print(f"Layer configuration: {self.configs[layer.name]}")
                print(f"Weight shapes:")
                print(f"  W_x: {weights[0].shape}")
                print(f"  W_h: {weights[1].shape}")
                print(f"  b: {weights[2].shape}")
                
            elif layer.name == 'dense':
                print(f"\nExtracting weights for {layer.name}")
                weights = layer.get_weights()
                
                self.weights[layer.name] = {
                    'kernel': weights[0],  # W
                    'bias': weights[1]  # b
                }
                
                self.configs[layer.name] = {
                    'units': layer.units,
                    'activation': layer.activation.__name__,
                    'use_bias': layer.use_bias
                }
                
                print(f"Layer configuration: {self.configs[layer.name]}")
                print(f"Weight shapes:")
                print(f"  W: {weights[0].shape}")
                print(f"  b: {weights[1].shape}")
    
    def apply_activation(self, x, activation_name):
        if activation_name not in self.activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
        return self.activation_functions[activation_name](x)
    
    def forward_pass(self, inputs):
        print("\nPerforming forward pass...")
        print(f"Input shape: {inputs.shape}")
        
        # Inisialisasi variabel
        batch_size = inputs.shape[0]
        h_states = {}
        
        # Proses setiap layer SimpleRNN
        for layer_name, layer_config in self.configs.items():
            if layer_name.startswith('simple_rnn'):
                print(f"\nProcessing {layer_name}")
                units = layer_config['units']
                
                # Inisialisasi hidden state
                h = np.zeros((batch_size, units))
                h_states[layer_name] = h
                
                # Mengambil bobot dari layer
                W_x = self.weights[layer_name]['kernel']
                W_h = self.weights[layer_name]['recurrent_kernel']
                b = self.weights[layer_name]['bias']
                
                # Memproses input sequence
                sequence_length = inputs.shape[1]
                outputs = []
                
                for t in range(sequence_length):
                    # Mendapatkan input pada waktu t
                    x_t = inputs[:, t, :]
                    
                    # Menghitung hidden state
                    h = self.apply_activation(
                        np.dot(x_t, W_x) + np.dot(h, W_h) + b,
                        self.hidden_activation
                    )
                    
                    # Menyimpan output jika return_sequences adalah True
                    if layer_config['return_sequences']:
                        outputs.append(h)
                    
                    # Mengupdate hidden state
                    h_states[layer_name] = h
                
                # Jika return_sequences adalah False, hidden state terakhir digunakan sebagai output
                if layer_config['return_sequences']:
                    outputs = np.stack(outputs, axis=1)
                    inputs = outputs
                else:
                    inputs = h
        
        # Memproses layer Dense
        dense_layer = next((name for name in self.configs.keys() if name == 'dense'), None)
        if dense_layer:
            print("\nProcessing dense layer")
            W = self.weights[dense_layer]['kernel']
            b = self.weights[dense_layer]['bias']

            dense_net = np.dot(inputs, W) + b

            predictions = self.apply_activation(dense_net, 'softmax')
            
            print(f"Output shape: {predictions.shape}")
            return predictions
        
        return inputs

def load_and_preprocess_test_data(data_path):
    print("Loading and preprocessing test data...")
    # Load test data
    test_df = pd.read_csv(data_path)
    
    labels = sorted(test_df['label'].unique())
    label_map = {label: i for i, label in enumerate(labels)}
    
    # Konversi label ke ID
    test_df['label_id'] = test_df['label'].map(label_map)
    
    return test_df, label_map

def preprocess_text(texts, max_sequence_length=100):
    print("Manually preprocessing text data...")
    
    # Simple tokenization 
    tokenized_texts = [text.lower().split() for text in texts]
    
    # Mapping token ke indeks
    all_words = set()
    for tokens in tokenized_texts:
        all_words.update(tokens)
    
    # Menambahkan spesial tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(all_words, start=2):
        vocab[word] = i
    
    # Konversi teks ke indeks
    sequences = []
    for tokens in tokenized_texts:
        sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_sequence_length]]
        if len(sequence) < max_sequence_length:
            sequence = sequence + [vocab["<PAD>"]] * (max_sequence_length - len(sequence))
        sequences.append(sequence)
    
    return np.array(sequences)

def compare_predictions(keras_model, scratch_model, test_data):
    print("\nComparing predictions between Keras model and scratch implementation...")
    test_texts = test_data['text'].values
    test_labels = test_data['label_id'].values
    
    # Preprocess text data 
    vectorized_texts = preprocess_text(test_texts)
    
    # Prediksi dengan keras  
    print("\nGenerating predictions from Keras model...")
    # Ekstrak embedding layer dari Keras model
    embedding_layer = keras_model.layers[1]  
    
    # Membuat model baru dengan input yang sesuai
    inputs = tf.keras.Input(shape=(vectorized_texts.shape[1],), dtype=tf.int32)
    x = embedding_layer(inputs)
    
    # Menambahkan layer SimpleRNN dan Dense dari Keras model
    for layer in keras_model.layers[2:]:
        x = layer(x)
    
    # Buat model baru
    preprocessed_model = tf.keras.Model(inputs, x)
    
    # Prediksi
    keras_preds = preprocessed_model.predict(vectorized_texts)
    keras_pred_classes = np.argmax(keras_preds, axis=1)
    
    # Prediksi dengan implementasi dari scratch
    print("\nGenerating predictions from scratch implementation...")
    scratch_preds = scratch_model.forward_pass(vectorized_texts)
    
    num_classes = len(np.unique(test_labels))
    scratch_pred_classes = np.argmax(scratch_preds, axis=1) % num_classes
    
    print("\n===== Prediction Distribution =====")
    keras_class_counts = np.bincount(keras_pred_classes, minlength=num_classes)
    scratch_class_counts = np.bincount(scratch_pred_classes, minlength=num_classes)
    true_class_counts = np.bincount(test_labels, minlength=num_classes)
    
    print("Class counts in true labels:")
    for i, count in enumerate(true_class_counts):
        print(f"  Class {i}: {count}")
    
    print("\nClass counts in Keras predictions:")
    for i, count in enumerate(keras_class_counts):
        print(f"  Class {i}: {count}")
    
    print("\nClass counts in Scratch predictions:")
    for i, count in enumerate(scratch_class_counts):
        print(f"  Class {i}: {count}")
    
    # Kalkulasi Evaluasi
    print("\nCalculating evaluation metrics...")
    keras_f1 = f1_score(test_labels, keras_pred_classes, average='macro')
    scratch_f1 = f1_score(test_labels, scratch_pred_classes, average='macro')
    keras_accuracy = np.mean(keras_pred_classes == test_labels)
    scratch_accuracy = np.mean(scratch_pred_classes == test_labels)

    
    # Summary
    print("\n~~~~ Model Comparison ~~~~")
    print(f"Keras Model Macro F1: {keras_f1:.4f}")
    print(f"Scratch Model Macro F1: {scratch_f1:.4f}")
    print(f"Keras Model Accuracy: {keras_accuracy:.4f}")
    print(f"Scratch Model Accuracy: {scratch_accuracy:.4f}")

    print("~~~~~~~~~~~~~~~~~~~~~~~=~~~\n")
    
    # Label names untuk classification report
    label_names = list(sorted(test_data['label'].unique()))
    
    # Print classification reports
    print("Keras Model Classification Report:")
    print(classification_report(test_labels, keras_pred_classes, target_names=label_names))
    
    print("\nScratch Model Classification Report:")
    print(classification_report(test_labels, scratch_pred_classes, target_names=label_names))
    
    return {
        'keras_f1': keras_f1,
        'scratch_f1': scratch_f1,
        'keras_accuracy': keras_accuracy,
        'scratch_accuracy': scratch_accuracy,
    }

def visualize_comparison(metrics):
    os.makedirs('images', exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    metrics_to_plot = ['f1', 'accuracy']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    keras_values = [metrics['keras_f1'], metrics['keras_accuracy']]
    scratch_values = [metrics['scratch_f1'], metrics['scratch_accuracy']]
    
    plt.bar(x - width/2, keras_values, width, label='Keras Model')
    plt.bar(x + width/2, scratch_values, width, label='Scratch Model')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Comparison of Keras and Simple RNN From Scratch')
    plt.xticks(x, ['Macro F1', 'Accuracy'])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.close()

def find_best_model():
    print("Finding the best performing model from experiment results...")
    results_path = 'results/experiment_results.json'
    if not os.path.exists(results_path):
        print("Error: Experiment results not found. Please run the experiments first.")
        return None
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Mencari best model berdasarkan F1-Macro
    best_f1 = 0
    best_config = None
    
    for experiment, configs in results.items():
        for config, metrics in configs.items():
            if metrics['f1_macro'] > best_f1:
                best_f1 = metrics['f1_macro']
                best_config = f"{experiment}_{config}"
    
    if best_config is None:
        print("Error: Could not determine best model from results.")
        return None
    
    # Mapping best configuration 
    if 'rnn_layers' in best_config:
        model_name = f"rnn_layers_{best_config.split('_')[-1]}"
    elif 'rnn_units' in best_config:
        model_name = f"rnn_units_{best_config.split('_')[-1]}"
    elif 'rnn_direction' in best_config:
        direction = best_config.split('_')[-1]
        model_name = f"rnn_{direction}"
    else:
        print("Error: Unrecognized model configuration.")
        return None
    
    model_path = f"models/{model_name}.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    
    print(f"Best model: {model_name} with F1-Macro = {best_f1:.4f}")
    return model_path

def main():
    print("~~~~ Simple RNN Forward Propagation from Scratch ~~~~")
    
    hidden_activation = 'tanh'  
    output_activation = 'tanh' 
    
    print(f"Using hidden activation: {hidden_activation}, output activation: {output_activation}")
    
    # Mencari data best model
    best_model_path = find_best_model()
    if best_model_path is None:
        print("Could not find the best model. Using default model path.")
        default_model_path = "models/rnn_layers_2.h5" 
        base_model_path = default_model_path
    
    # Load model keras
    print("\nLoading Keras model...")
    keras_model = load_model(best_model_path)
    
    # Inisialisasi the scratch implementation
    scratch_model = RNNFromScratch(
        best_model_path, 
        hidden_activation=hidden_activation,
        output_activation=output_activation
    )
    
    # Load test data
    test_data, _ = load_and_preprocess_test_data('data/test.csv')
    
    # Compare predictions
    metrics = compare_predictions(keras_model, scratch_model, test_data)
    
    # Visualize comparison
    visualize_comparison(metrics)
    
    print("\n~~~~ Process Completed! ~~~~")

if __name__ == "__main__":
    main() 