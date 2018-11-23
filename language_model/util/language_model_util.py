import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["create_variable_initializer", "create_weight_regularizer", "create_activation_function",
           "softmax_with_mask", "generate_masked_data", "generate_onehot_label", "align_sequence", "reverse_sequence"]

def create_variable_initializer(initializer_type,
                                random_seed=None,
                                data_type=tf.float32):
    """create variable initializer"""
    if initializer_type == "zero":
        initializer = tf.zeros_initializer
    elif initializer_type == "orthogonal":
        initializer = tf.orthogonal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "random_uniform":
        initializer = tf.random_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_uniform":
        initializer = tf.glorot_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "xavier_uniform":
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=random_seed, dtype=tf.float32)
    elif initializer_type == "random_normal":
        initializer = tf.random_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "truncated_normal":
        initializer = tf.truncated_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_normal":
        initializer = tf.glorot_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "xavier_normal":
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=random_seed, dtype=tf.float32)
    else:
        initializer = None
    
    return initializer

def create_weight_regularizer(regularizer_type,
                              scale):
    """create weight regularizer"""
    if regularizer_type == "l1":
        regularizer = tf.contrib.layers.l1_regularizer(scale)
    elif regularizer_type == "l2":
        regularizer = tf.contrib.layers.l2_regularizer(scale)
    else:
        regularizer = None
    
    return regularizer

def create_activation_function(activation):
    """create activation function"""
    if activation == "relu":
        activation_function = tf.nn.relu
    elif activation == "relu6":
        activation_function = tf.nn.relu6
    elif activation == "leaky_relu":
        activation_function = tf.nn.leaky_relu
    elif activation == "elu":
        activation_function = tf.nn.elu
    elif activation == "crelu":
        activation_function = tf.nn.crelu
    elif activation == "selu":
        activation_function = tf.nn.selu
    elif activation == "gelu":
        activation_function = gelu
    elif activation == "tanh":
        activation_function = tf.nn.tanh
    elif activation == "sigmoid":
        activation_function = tf.nn.sigmoid
    elif activation == "softplus":
        activation_function = tf.nn.softplus
    else:
        activation_function = None
    
    return activation_function

def softmax_with_mask(input_data,
                      input_mask,
                      axis=-1):
    """compute softmax with masking"""    
    return tf.nn.softmax(input_data + MIN_FLOAT * (1 - input_mask), axis=axis)

def generate_masked_data(input_data,
                         input_mask):
    """generate masked data"""
    return input_data + MIN_FLOAT * (1 - input_mask)

def generate_onehot_label(input_data,
                          input_depth):
    """generate one-hot label"""
    return tf.one_hot(input_data, depth=input_depth, on_value=1.0, off_value=0.0, dtype=tf.float32)

def align_sequence(input_data,
                   input_mask,
                   alignment):
    """align sequence"""
    input_data_shape = tf.shape(input_data)
    input_mask_shape = tf.shape(input_mask)
    shape_size = len(input_data.get_shape().as_list())
    if shape_size > 3:
        input_data = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-2:]], axis=0))
        input_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-2:]], axis=0))
    
    if alignment > 0:
        padding = tf.constant([[0, 0], [0, alignment], [0, 0]])
        output_mask = tf.pad(input_mask[:,alignment:,:], padding)
        output_data = tf.pad(input_data[:,alignment:,:], padding) * output_mask
    else:
        output_data = input_data
        output_mask = input_mask
    
    if shape_size > 3:
        output_data_shape = tf.shape(output_data)
        output_mask_shape = tf.shape(output_mask)
        output_data = tf.reshape(output_data,
            shape=tf.concat([input_data_shape[:-2], output_data_shape[-2:]], axis=0))
        output_mask = tf.reshape(output_mask,
            shape=tf.concat([input_mask_shape[:-2], output_mask_shape[-2:]], axis=0))
    
    return output_data, output_mask

def reverse_sequence(input_data,
                     input_mask):
    """reverse sequence"""
    input_data_shape = tf.shape(input_data)
    input_mask_shape = tf.shape(input_mask)
    shape_size = len(input_data.get_shape().as_list())
    if shape_size > 3:
        input_data = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-2:]], axis=0))
        input_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-2:]], axis=0))
    
    input_length = tf.cast(tf.reduce_sum(tf.squeeze(input_mask, axis=-1), axis=-1), dtype=tf.int32)
    output_data = tf.reverse_sequence(input_data, input_length, seq_axis=1, batch_axis=0)
    output_mask = input_mask
    
    if shape_size > 3:
        output_data_shape = tf.shape(output_data)
        output_mask_shape = tf.shape(output_mask)
        output_data = tf.reshape(output_data,
            shape=tf.concat([input_data_shape[:-2], output_data_shape[-2:]], axis=0))
        output_mask = tf.reshape(output_mask,
            shape=tf.concat([input_mask_shape[:-2], output_mask_shape[-2:]], axis=0))
    
    return output_data, output_mask

def gelu(input_tensor):
    """Gaussian Error Linear Unit"""
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf
