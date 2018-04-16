import collections

import numpy as np
import tensorflow as tf

from model.language_model import *
from model.language_model_bidirectional import *
from util.default_util import *
from util.data_util import *
from util.language_model_util import *

__all__ = ["TrainModel", "EvalModel", "InferModel", "EncodeModel",
           "create_train_model", "create_eval_model", "create_infer_model", "create_encode_model",
           "get_model_creator", "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "embedding"))):
    pass

class EvalModel(collections.namedtuple("EvalModel",
    ("graph", "model", "data_pipeline", "embedding"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "input_data", "embedding"))):
    pass

class EncodeModel(collections.namedtuple("EncodeModel",
    ("graph", "model", "data_pipeline", "input_data", "embedding"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, embedding_data, vocab_size, vocab_index,
            vocab_inverted_index) = prepare_data(logger, hyperparams.data_train_file,
            hyperparams.data_vocab_file, hyperparams.data_embedding_file, hyperparams.data_full_embedding_file,
            hyperparams.data_vocab_size, hyperparams.model_embed_dim, hyperparams.data_unk, hyperparams.data_sos,
            hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create train data pipeline")
        data_pipeline = create_data_pipeline(hyperparams.data_train_file, vocab_index,
            hyperparams.data_max_length, hyperparams.data_sos, hyperparams.data_eos,
            hyperparams.data_pad, hyperparams.train_batch_size, hyperparams.train_random_seed,
            hyperparams.train_enable_shuffle, hyperparams.model_type)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index, vocab_inverted_index=vocab_inverted_index,
            mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline, embedding=embedding_data)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare eval data")
        (input_data, embedding_data, vocab_size, vocab_index,
            vocab_inverted_index) = prepare_data(logger, hyperparams.data_eval_file,
            hyperparams.data_vocab_file, hyperparams.data_embedding_file, hyperparams.data_full_embedding_file,
            hyperparams.data_vocab_size, hyperparams.model_embed_dim, hyperparams.data_unk, hyperparams.data_sos,
            hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create eval data pipeline")
        data_pipeline = create_data_pipeline(hyperparams.data_eval_file, vocab_index,
            hyperparams.data_max_length, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_eval_batch_size, hyperparams.train_random_seed, False, hyperparams.model_type)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index, vocab_inverted_index=vocab_inverted_index,
            mode="eval", scope=hyperparams.model_scope)
        
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline, embedding=embedding_data)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare infer data")
        (input_data, embedding_data, vocab_size, vocab_index,
            vocab_inverted_index) = prepare_data(logger, hyperparams.data_eval_file,
            hyperparams.data_vocab_file, hyperparams.data_embedding_file, hyperparams.data_full_embedding_file,
            hyperparams.data_vocab_size, hyperparams.model_embed_dim, hyperparams.data_unk, hyperparams.data_sos,
            hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create infer data pipeline")
        data_pipeline = create_dynamic_data_pipeline(vocab_index, hyperparams.data_max_length,
            hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_type)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index, vocab_inverted_index=vocab_inverted_index,
            mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline,
            input_data=input_data, embedding=embedding_data)

def create_encode_model(logger,
                        hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare encoding data")
        (input_data, embedding_data, vocab_size, vocab_index,
            vocab_inverted_index) = prepare_data(logger, hyperparams.data_eval_file,
            hyperparams.data_vocab_file, hyperparams.data_embedding_file, hyperparams.data_full_embedding_file,
            hyperparams.data_vocab_size, hyperparams.model_embed_dim, hyperparams.data_unk, hyperparams.data_sos,
            hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create encoding data pipeline")
        data_pipeline = create_dynamic_data_pipeline(vocab_index, hyperparams.data_max_length,
            hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_type)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index, vocab_inverted_index=vocab_inverted_index,
            mode="encode", scope=hyperparams.model_scope)
        
        return EncodeModel(graph=graph, model=model, data_pipeline=data_pipeline,
            input_data=input_data, embedding=embedding_data)

def get_model_creator(model_type):
    if model_type == "forward_only":
        model_creator = LanguageModel
    elif model_type == "bi_directional":
        model_creator = LanguageModelBidirectional
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model):
    with model.graph.as_default():
        model.model.restore(sess)
