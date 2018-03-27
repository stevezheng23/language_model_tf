import numpy as np
import tensorflow as tf

__all__ = ["create_embedding", "create_activation_function"]

def create_embedding(vocab_size,
                     embedding_dim):
    """create embedding layer"""
    init_width = 0.5 / embedding_dim
    embed_initializer = tf.random_uniform_initializer(-init_width, init_width, dtype=tf.float32)
    embedding = tf.get_variable("embedding", shape=[vocab_size, embedding_dim],
        initializer=embed_initializer, dtype=tf.float32, trainable=True)
    
    return embedding

def create_activation_function(activation):
    """create activation function"""
    if activation == "tanh":
        activation_function = tf.nn.tanh
    elif activation == "relu":
        activation_function = tf.nn.relu
    elif activation == "leaky_relu":
        activation_function = tf.nn.leaky_relu
    elif activation == "sigmoid":
        activation_function = tf.nn.sigmoid
    else:
        activation_function = None
    
    return activation_function
