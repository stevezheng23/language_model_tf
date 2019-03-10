import collections

import numpy as np
import tensorflow as tf

from model.seq_lm import *
from util.data_util import *

__all__ = ["TrainModel", "EvalModel", "DecodeModel", "EncodeModel",
           "create_train_model", "create_eval_model", "create_decode_model", "create_encode_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "input_data", "word_embedding"))):
    pass

class EvalModel(collections.namedtuple("EvalModel",
    ("graph", "model", "data_pipeline", "input_data", "word_embedding"))):
    pass

class DecodeModel(collections.namedtuple("DecodeModel",
    ("graph", "model", "data_pipeline", "input_data", "word_embedding"))):
    pass

class EncodeModel(collections.namedtuple("EncodeModel",
    ("graph", "model", "data_pipeline", "input_data", "word_embedding"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_train_file, hyperparams.data_word_vocab_file, hyperparams.data_word_vocab_size,
            hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim, hyperparams.data_embedding_file,
            hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad, hyperparams.data_word_sos,
            hyperparams.data_word_eos, hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained,
            hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
            hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_large_file_train)
        
        data_size = len(input_data) if input_data is not None else None
        external_data = {}
        
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create train text dataset")
            text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
                hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_text_word_dataset, input_text_char_dataset,
                word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
                hyperparams.model_char_feat_enable, hyperparams.train_random_seed, hyperparams.train_enable_shuffle,
                hyperparams.train_shuffle_buffer_size, text_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data
            
            logger.log_print("# create train text dataset")
            text_dataset = get_text_dataset(hyperparams.data_train_file)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
                hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train data pipeline")
            data_pipeline = create_data_pipeline(input_text_word_dataset, input_text_char_dataset,
                word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
                hyperparams.model_char_feat_enable, hyperparams.train_random_seed, hyperparams.train_enable_shuffle,
                hyperparams.train_shuffle_buffer_size, data_size, hyperparams.train_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="train", external_data=external_data, scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            input_data=input_data, word_embedding=word_embed_data)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare eval data")
        (input_data, word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_eval_file, hyperparams.data_word_vocab_file, hyperparams.data_word_vocab_size,
            hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim, hyperparams.data_embedding_file,
            hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad, hyperparams.data_word_sos,
            hyperparams.data_word_eos, hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained,
            hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
            hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_char_feat_enable, False)
        
        data_size = len(input_data) if input_data is not None else None
        external_data = {}
        
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create eval text dataset")
            text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
                hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create eval data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_text_word_dataset, input_text_char_dataset,
                word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
                hyperparams.model_char_feat_enable, hyperparams.train_random_seed, False, 0,
                text_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data
            
            logger.log_print("# create eval text dataset")
            text_dataset = get_text_dataset(hyperparams.data_eval_file)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
                hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create eval data pipeline")
            data_pipeline = create_data_pipeline(input_text_word_dataset, input_text_char_dataset,
                word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
                hyperparams.model_char_feat_enable, hyperparams.train_random_seed, False, 0,
                data_size, hyperparams.train_eval_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="eval", external_data=external_data, scope=hyperparams.model_scope)
        
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline,
            input_data=input_data, word_embedding=word_embed_data)

def create_decode_model(logger,
                        hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare decode data")
        (input_data, word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_eval_file, hyperparams.data_word_vocab_file, hyperparams.data_word_vocab_size,
            hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim, hyperparams.data_embedding_file,
            hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad, hyperparams.data_word_sos,
            hyperparams.data_word_eos, hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained,
            hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
            hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_char_feat_enable, False)
        
        logger.log_print("# create decode text dataset")
        text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
        input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
            word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
            hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
            hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

        logger.log_print("# create decode data pipeline")
        data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        data_pipeline = create_dynamic_pipeline(input_text_word_dataset, input_text_char_dataset,
            word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
            hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
            hyperparams.model_char_feat_enable, hyperparams.train_random_seed, False, 0,
            text_placeholder, data_size_placeholder, batch_size_placeholder)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="decode", external_data={}, scope=hyperparams.model_scope)
        
        return DecodeModel(graph=graph, model=model, data_pipeline=data_pipeline,
            input_data=input_data, word_embedding=word_embed_data)

def create_encode_model(logger,
                        hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare encode data")
        (input_data, word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_data(logger,
            hyperparams.data_eval_file, hyperparams.data_word_vocab_file, hyperparams.data_word_vocab_size,
            hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim, hyperparams.data_embedding_file,
            hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad, hyperparams.data_word_sos,
            hyperparams.data_word_eos, hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained,
            hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
            hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_char_feat_enable, False)
        
        data_size = len(input_data) if input_data is not None else None
        external_data = {}
        
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create encode text dataset")
            text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
                hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create encode data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_text_word_dataset, input_text_char_dataset,
                word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
                hyperparams.model_char_feat_enable, hyperparams.train_random_seed, False, 0,
                text_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data
            
            logger.log_print("# create encode text dataset")
            text_dataset = get_text_dataset(hyperparams.data_eval_file)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_max_word_size, hyperparams.data_word_pad, hyperparams.data_word_sos,
                hyperparams.data_word_eos, hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_max_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create encode data pipeline")
            data_pipeline = create_data_pipeline(input_text_word_dataset, input_text_char_dataset,
                word_vocab_size, word_vocab_index, word_vocab_inverted_index, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_char_pad,
                hyperparams.model_char_feat_enable, hyperparams.train_random_seed, False, 0,
                data_size, hyperparams.train_eval_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="encode", external_data=external_data, scope=hyperparams.model_scope)
        
        return EncodeModel(graph=graph, model=model, data_pipeline=data_pipeline,
            input_data=input_data, word_embedding=word_embed_data)

def get_model_creator(model_type):
    if model_type == "seq_lm":
        model_creator = SequenceLM
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model,
               ckpt_file,
               ckpt_type):
    with model.graph.as_default():
        model.model.restore(sess, ckpt_file, ckpt_type)
