import collections

import numpy as np
import tensorflow as tf

from model.language_model import *
from util.default_util import *
from util.data_util import *
from util.representation_util import *

__all__ = ["TrainModel", "EvalModel", "create_train_model", "create_eval_model"
           "get_model_creator", "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "embedding"))):
    pass

class EvalModel(collections.namedtuple("EvalModel",
    ("graph", "model", "data_pipeline", "embedding"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, embedding_data, src_vocab_size, vocab_index,
            vocab_inverted_index) = prepare_data(logger, hyperparams.data_train_file,
            hyperparams.data_vocab_file, hyperparams.data_embedding_file, hyperparams.data_full_embedding_file,
            hyperparams.data_vocab_size, hyperparams.model_embed_dim, hyperparams.data_unk, hyperparams.data_sos,
            hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create train data pipeline")
        data_pipeline = create_lm_pipeline(hyperparams.data_train_file, vocab_index,
            hyperparams.data_max_length, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_batch_size, hyperparams.train_random_seed)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index, vocab_inverted_index=vocab_inverted_index, mode="train",
            pretrained_embedding=hyperparams.model_pretrained_embedding, scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline, embedding=embedding_data)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare evaluation data")
        (input_data, embedding_data, src_vocab_size, vocab_index,
            vocab_inverted_index) = prepare_data(logger, hyperparams.data_train_file,
            hyperparams.data_vocab_file, hyperparams.data_embedding_file, hyperparams.data_full_embedding_file,
            hyperparams.data_vocab_size, hyperparams.model_embed_dim, hyperparams.data_unk, hyperparams.data_sos,
            hyperparams.data_eos, hyperparams.data_pad, hyperparams.model_pretrained_embedding)
    
        logger.log_print("# create evaluation data pipeline")
        data_pipeline = create_lm_pipeline(hyperparams.data_train_file, vocab_index,
            hyperparams.data_max_length, hyperparams.data_sos, hyperparams.data_eos, hyperparams.data_pad,
            hyperparams.train_batch_size, hyperparams.train_random_seed)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            vocab_size=vocab_size, vocab_index=vocab_index, vocab_inverted_index=vocab_inverted_index, mode="eval",
            pretrained_embedding=hyperparams.model_pretrained_embedding, scope=hyperparams.model_scope)
        
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline, embedding=embedding_data)

def get_model_creator(model_type):
    if model_type == "vanilla":
        model_creator = LanguageModel
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(hyperparams.model_type))
    
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
