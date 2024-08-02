import numpy as np

# Seed for reproducibility
np.random.seed(42)

def initialize_network(input_dim, hidden_layers, output_dim):
    np.random.seed(42)  # For consistent results
    layers = [input_dim] + hidden_layers + [output_dim] # Where Input and Output could be Nx1 vectors
    weights = {}
    biases = {}

    for i in range(1, len(layers)):
        weights[i] = np.random.randn(layers[i-1], layers[i]) * 0.01
        biases[i] = np.zeros((1, layers[i]))
    
    return weights, biases

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def forward_propagation(X,weights,biases):
    activations = {0:X}
    pre_activations = {}
    layers = len(weights)
    for i in range(1, layers + 1):
        computed_activation = activations[i - 1].dot(weights[i]) + biases[i]
        pre_activations[i - 1] = computed_activation
        '''
            Assuming we're training neural network for binary classification,
            the best choice for activation function would be:
                - Relu For hidden layer
                - Sigmoid for output layer
        '''
        if i == layers:
            activations[i] = sigmoid(computed_activation)
        else:
            activations[i] = relu(computed_activation)
    
    return activations, pre_activations

# full binary cross-entropy loss
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

def backpropagation(y_true, y_pred, activations, pre_activations, weights):
    gradients = {}
    m = y_true.shape[0]
    layers = len(weights)

    # Output layer error
    delta = y_pred - y_true
    gradients["dW" + str(layers)] = activations[layers-1].T.dot(delta) / m
    gradients["db" + str(layers)] = np.sum(delta, axis=0, keepdims=True) / m

    # Propagate the error backwards
    for i in range(layers-1, 0, -1):
        delta = delta.dot(weights[i+1].T) * relu_derivative(pre_activations[i])
        gradients["dW" + str(i)] = activations[i-1].T.dot(delta) / m
        gradients["db" + str(i)] = np.sum(delta, axis=0, keepdims=True) / m
    
    return gradients

def update_parameters(weights, biases, gradients, lr=0.01):
    layers = len(weights)
    for i in range(1, layers + 1):
        weights[i] -= lr * gradients["dW" + str(i)]
        biases[i] -= lr * gradients["db" + str(i)]
    
    return weights, biases

def create_mini_batches(X, y, batch_size):
    m = X.shape[0]
    mini_batches = []
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]

    num_complete_batches = m // batch_size
    for i in range(num_complete_batches):
        X_batch = shuffled_X[i*batch_size:(i+1)*batch_size]
        y_batch = shuffled_y[i*batch_size:(i+1)*batch_size]
        mini_batches.append((X_batch, y_batch))
    
    if m % batch_size != 0:
        X_batch = shuffled_X[num_complete_batches*batch_size:]
        y_batch = shuffled_y[num_complete_batches*batch_size:]
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches

def train(X_train, y_train, epochs, batch_size, lr, input_dim, hidden_layers, output_dim):
    weights, biases = initialize_network(input_dim, hidden_layers, output_dim)
    
    for epoch in range(epochs):
        mini_batches = create_mini_batches(X_train, y_train, batch_size)
        for X_batch, y_batch in mini_batches:
            activations, pre_activations = forward_propagation(X_batch, weights, biases)
            loss = compute_loss(y_batch, activations[len(weights)])
            gradients = backpropagation(y_batch, activations[len(weights)], activations, pre_activations, weights, biases)
            weights, biases = update_parameters(weights, biases, gradients, lr)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    return weights, biases

def predict(X, weights, biases):
    activations, _ = forward_propagation(X, weights, biases)
    return activations[len(weights)]



    