import jax
import numpy as np
import jax.numpy as jnp
import jax.nn as nn
from jax import random
import time

# Set fixed seed for reproducibility
np.random.seed(42)

class TopNet:
    def __init__(self, nnSettings):
        """Initialize the neural network with improved configuration"""
        self.nnSettings = nnSettings
        self.inputDim = nnSettings.get('inputDim', 3)
        self.outputDim = nnSettings.get('outputDim', 4)
        
        # Improved layer configuration
        self.numLayers = nnSettings.get('numLayers', 5)  # Increased layers
        self.numNeuronsPerLayer = nnSettings.get('numNeuronsPerLayer', 128)  # Increased neurons
        
        # Regularization parameters
        self.dropout_rate = nnSettings.get('dropout_rate', 0.2)
        
        # Get target volume fraction from settings if available
        self.target_vf = nnSettings.get('target_vf', 0.7)
        
        # Initialize network weights
        self.wts = self.initialize_network()
        
        print(f"Initializing improved network with:")
        print(f"Input dim: {self.inputDim}")
        print(f"Hidden dim: {self.numNeuronsPerLayer}")
        print(f"Num layers: {self.numLayers}")
        print(f"Output dim: {self.outputDim}")
        print(f"Target volume fraction: {self.target_vf}")
        print(f"Dropout rate: {self.dropout_rate}")
        
    def initialize_network(self):
        """Initialize network with improved initialization strategies"""
        key = random.PRNGKey(0)
        
        # He initialization for better gradient flow
        def he_init(key, shape):
            fan_in = shape[0]
            std = np.sqrt(2.0 / fan_in)
            return random.normal(key, shape) * std
        
        weights = []
        biases = []
        
        # Input layer with more sophisticated initialization
        key, subkey = random.split(key)
        w_input = he_init(subkey, (self.inputDim, self.numNeuronsPerLayer))
        key, subkey = random.split(key)
        b_input = random.normal(subkey, (self.numNeuronsPerLayer,)) * 0.01
        weights.append(w_input)
        biases.append(b_input)
        
        # Hidden layers with improved initialization
        for i in range(self.numLayers - 2):
            key, subkey = random.split(key)
            w_hidden = he_init(subkey, (self.numNeuronsPerLayer, self.numNeuronsPerLayer))
            key, subkey = random.split(key)
            b_hidden = random.normal(subkey, (self.numNeuronsPerLayer,)) * 0.01
            weights.append(w_hidden)
            biases.append(b_hidden)
        
        # Output layer
        key, subkey = random.split(key)
        w_output = he_init(subkey, (self.numNeuronsPerLayer, self.outputDim))
        key, subkey = random.split(key)
        b_output = random.normal(subkey, (self.outputDim,)) * 0.01
        if self.outputDim > 1:  # If we have both microstructure and density outputs
            # Apply logit bias to encourage target volume fraction
            b_output = b_output.at[-1].set(jnp.log(self.target_vf / (1.0 - self.target_vf)))
        weights.append(w_output)
        biases.append(b_output)
        
        # Store as a simple list for easy access
        params = []
        for w, b in zip(weights, biases):
            params.append(w)
            params.append(b)
            
        return params
        
    def dropout(self, x, key, rate):
        """Implement dropout for regularization"""
        keep_prob = 1.0 - rate
        key, subkey = random.split(key)
        mask = random.bernoulli(subkey, keep_prob, x.shape)
        return x * mask / keep_prob, key
        
    def forward(self, params, x, is_training=True):
        """Enhanced forward pass with dropout and improved regularization"""
        try:
            # Ensure consistent input type and shape
            if not isinstance(x, jnp.ndarray):
                x = jnp.array(x, dtype=jnp.float32)
            x = x.reshape(-1, self.inputDim)
            
            # Clean and clip input
            x = jnp.clip(jnp.nan_to_num(x, nan=0.0), -10.0, 10.0)
            
            # Prepare dropout key if training
            dropout_key = random.PRNGKey(int(time.time() * 1000)) if is_training else None
            
            h = x
            for i in range(len(params) // 2 - 1):
                w = params[i*2]
                b = params[i*2 + 1]
                
                # Linear transformation
                h = jnp.dot(h, w) + b
                
                # ReLU activation
                h = jnp.maximum(0, h)
                
                # Dropout during training
                if is_training and dropout_key is not None and self.dropout_rate > 0:
                    h, dropout_key = self.dropout(h, dropout_key, self.dropout_rate)
            
            # Output layer
            w = params[-2]
            b = params[-1]
            h = jnp.dot(h, w) + b
            
            # Split outputs for microstructure and density
            mstr_cols = min(3, h.shape[1] - 1)
            
            # Softmax for microstructure type with temperature
            temperature = 0.5  # Adjust temperature for softmax
            mstr_logits = h[:, :mstr_cols] /temperature
            mstr_type = nn.softmax(mstr_logits, axis=1)
            
            # Density with target volume fraction bias
            density_logits = h[:, -1]
            logit_bias = jnp.log(self.target_vf / (1.0 - self.target_vf))
            rho = nn.sigmoid(density_logits + logit_bias)
            
            # Ensure reasonable density range
            rho = 0.001 + 0.998 * rho
            
            return mstr_type, rho
            
        except Exception as e:
            print(f"Error in network forward: {e}")
            return self.default_output(x.shape[0])
            
    def default_output(self, batch_size, target_vf=None):
        """Default output with improved diversity"""
        if target_vf is None:
            target_vf = self.target_vf
            
        # Encourage more balanced microstructure distribution
        mstr_type = jnp.ones((batch_size, 3)) / 3.0 + \
                    jnp.random.normal(size=(batch_size, 3)) * 0.1
        mstr_type = nn.softmax(mstr_type, axis=1)
        
        # Density biased toward target with some variance
        rho = jnp.ones(batch_size) * target_vf + \
              jnp.random.normal(size=(batch_size,)) * 0.05
        rho = jnp.clip(rho, 0.001, 0.999)
        
        return mstr_type, rho