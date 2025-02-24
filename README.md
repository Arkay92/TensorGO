# TensorGO: A Go-based Deep Learning Framework

TensorGO is a lightweight, modular deep learning framework written in Go. It provides GPU acceleration using Gorgonia, supports multiple activation functions, optimizers, and loss functions, and includes both single-threaded and multi-threaded distributed training capabilities.

## Features

- **Fully Connected Neural Networks**
- **GPU Acceleration with Gorgonia**
- **Support for Activation Functions:** Sigmoid, ReLU, Leaky ReLU
- **Loss Functions:** Mean Squared Error (MSE), Cross-Entropy
- **Optimizers:** Adam, Stochastic Gradient Descent (SGD)
- **Distributed Multi-threaded Training**
- **Model Checkpointing & Loading**
- **Graph Visualization using GraphViz (DOT format)**

## Installation

1. Install Go (if not already installed):
   ```sh
   sudo apt update && sudo apt install golang
   ```
   
2. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/TensorGO.git
   cd TensorGO
   ```
   
3. Install dependencies:
   ```sh
   go mod tidy
   ```

## Usage

### Running the Example XOR Model

To train a simple XOR model using the neural network implementation:

```sh
 go run main.go --gpu
```

### Example Code

```go
nn := NewNeuralNetwork(0.1, NewAdam(), CrossEntropyLoss{})

// Define the XOR network architecture
nn.AddLayer(2, 4, "leaky")
nn.AddLayer(4, 4, "leaky")
nn.AddLayer(4, 4, "leaky")
nn.AddLayer(4, 1, "sigmoid")

// XOR training dataset
inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
targets := [][]float64{{0}, {1}, {1}, {0}}

// Train the model
nn.Train(inputs, targets, 10000, 0)
```

### Enabling GPU Acceleration

Pass the `--gpu` flag when running the program to enable GPU acceleration with Gorgonia:

```sh
 go run main.go --gpu
```

### Model Saving & Loading

Save a trained model:

```go
err := nn.SaveModel("model.gob")
if err != nil {
    log.Fatalf("Failed to save model: %v", err)
}
```

Load a saved model:

```go
loadedModel, err := LoadModel("model.gob", NewAdam(), CrossEntropyLoss{})
if err != nil {
    log.Fatalf("Failed to load model: %v", err)
}
```

### Distributed Training

TensorGO supports distributed training using Go's concurrency features:

```go
nn.DistributedTrain(inputs, targets, 10000, 0, 4) // 4 worker threads
```

### Model Visualization

Generate a DOT file for GraphViz visualization:

```sh
 go run main.go
 dot -Tpng graph.dot -o graph.png
```

### Checkpointing

The model automatically saves checkpoints every 5000 epochs:

```sh
 checkpoints/
    checkpoint_epoch_0.gob
    checkpoint_epoch_5000.gob
    model_final.gob
```

## Performance

The XOR problem is learned quickly, with loss reaching near zero:

```
Epoch 0 - Loss: 0.213057
Epoch 1000 - Loss: 0.000000
Epoch 5000 - Loss: 0.000000
...
Predictions:
Input: [0 0], Output: [8.38e-22]
Input: [0 1], Output: [1]
Input: [1 0], Output: [1]
Input: [1 1], Output: [2.12e-11]
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
