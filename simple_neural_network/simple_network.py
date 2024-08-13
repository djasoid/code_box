import numpy as np
import struct

class Layer:
    """A layer in a neural network, only holds weights and biases."""
    def __init__(self, node_count, last_layer_node_count) -> None:
        self.node_count = node_count
        self.weights = np.random.randn(last_layer_node_count, node_count) * np.sqrt(1. / last_layer_node_count) # Xavier initialization
        self.bias = np.random.rand(node_count)
        
class Network:
    """A simple fully connected neural network, using the sigmoid activation function and mean squared error loss function."""
    def __init__(self, layer_counts: list) -> None:
        self.input_size = layer_counts[0]
        self.output_size = layer_counts[-1]
        self.layers = [Layer(layer_counts[i], layer_counts[i-1]) for i in range(1, len(layer_counts))]


    def forward(self, input_arr: np.ndarray) -> np.ndarray: # the input should match the input_size
        """Forward pass through the network for a single input."""
        for layer in self.layers:
            output_arr = np.dot(input_arr, layer.weights) + layer.bias
            output_arr = 1 / (1 + np.exp(-output_arr)) # sigmoid activation function
            input_arr = output_arr
        return output_arr

    def predict(self, input_arr: np.ndarray) -> int:
        """Predict the class of a single input."""
        return np.argmax(self.forward(input_arr))
    
    def loss(self, input_arr: np.ndarray, target: np.ndarray) -> float:
        """Calculate the mean squared error loss for the given inputs and targets."""
        output = self.forward(input_arr)
        return (np.mean((output - target) ** 2) / 2) # mean squared error loss

    def backward(self, input_arr: np.ndarray, target: np.ndarray, learning_rate: float) -> None:
        """Backward pass through the network for a single input, updating the weights and biases."""
        activations = [input_arr]
        activation = input_arr
        # Forward pass
        for layer in self.layers:
            linear_output = np.dot(activation, layer.weights) + layer.bias
            activation = 1 / (1 + np.exp(-linear_output))
            activations.append(activation)
        
        # calculate MSE cost
        loss = np.mean((activation - target) ** 2)
        
        # Backward pass
        d_loss = activation - target # MSE cost derivative dC/da
        delta = d_loss * (activations[-1] * (1 - activations[-1])) # sigmoid derivative dC/dz = dC/da * da/dz, da/dz = a * (1 - a)
        for i in range(len(self.layers) - 1, -1, -1): # iterate backwards through the layers
            layer = self.layers[i]
            d_weight = np.outer(activations[i], delta) # weight derivative dC/dw = dC/dz * dz/dw = dC/dz * a-1
            d_bias = delta # bias derivative dC/db = dC/dz * dz/db = dC/dz * 1
            if i > 0:#  dC/da-1 = dC/dz * dz/da-1 = dC/dz * w     da-1/dz-1 = a-1 * (1 - a-1)
                delta = np.dot(delta, self.layers[i].weights.T) * (activations[i] * (1 - activations[i]))
                # calculate the next delta using the current delta and the weights of the current layer sigmoid derivative 
                # dC/dz-1 = dC/da * da/dz * dz/da-1 * da-1/dz-1
            layer.weights -= learning_rate * d_weight # weight update
            layer.bias -= learning_rate * d_bias # bias update
        
        return loss
    
    def train(self, input_arrs: list[np.ndarray], targets: list[np.ndarray], learning_rate: float) -> None:
        """Train the network on a list of inputs and targets."""
        for i in range(len(input_arrs)):
            los = self.backward(input_arrs[i], targets[i], learning_rate)
            print(f'Training {i + 1}/{len(input_arrs)}, loss {los}', end='\r')

            
            
def load_mnist_images(filename: str) -> np.ndarray:
    """Load MNIST images from a file."""
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        f.seek(16)
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows * cols)
        return [images[i] for i in range(num_images)]

def load_mnist_labels(filename: str) -> np.ndarray:
    """Load MNIST labels from a file. (one-hot encoding)"""
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        f.seek(8)
        labels = np.fromfile(f, dtype=np.uint8)
        targets = np.zeros((num_labels, 10))
        for i in range(num_labels):
            targets[i][labels[i]] = 1
        return targets
    
def load_labels(filename: str) -> np.ndarray:
    """Load MNIST labels from a file."""
    with open(filename, 'rb') as f:
        f.seek(8)
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def model_test(model: Network, test_images: list[np.ndarray], test_labels: np.ndarray) -> None:
    """Test the model on a list of images and labels."""
    correct = 0
    for i in range(len(test_images)):
        output = model.forward(test_images[i])
        if np.argmax(output) == test_labels[i]:
            correct += 1
    print(f'Accuracy: {correct / len(test_images) * 100}%')

model = Network([28 * 28, 16, 16, 10])

train_images = load_mnist_images('mnist/train-images.idx3-ubyte')
train_labels = load_mnist_labels('mnist/train-labels.idx1-ubyte')
test_images = load_mnist_images('mnist/t10k-images.idx3-ubyte')
test_labels = load_labels('mnist/t10k-labels.idx1-ubyte')

# traning the model
epochs = 5
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    model.train(train_images, train_labels, 0.005)
    print()
    model_test(model, test_images, test_labels)
