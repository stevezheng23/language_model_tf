import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["DataPipeline", "create_data_pipeline",
           "load_vocab_table", "create_vocab_table", "create_vocab_file",
           "create_embedding_file", "load_input"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "input_data", "input_length", "input_data_placeholder", "batch_size_placeholder"))):
    pass

def create_data_pipeline(vocab_index,
                         max_length,
                         sos,
                         eos,
                         pad):
    """create source data pipeline based on config"""
    pad_id = tf.cast(vocab_index.lookup(tf.constant(pad)), tf.int32)
    
    input_data_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices(input_data_placeholder)
    dataset = dataset.map(lambda line: tf.string_split([line], delimiter=' ').values)
    dataset = dataset.map(lambda line: line[:max_length])
    
    dataset = dataset.map(lambda line: tf.cast(vocab_index.lookup(line), tf.int32))
    dataset = dataset.map(lambda line: (line, tf.size(line)))
    
    dataset = dataset.padded_batch(
        batch_size=batch_size_placeholder,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([])),
        padding_values=(
            pad_id,
            0))
    
    iterator = dataset.make_initializable_iterator()
    input_ids, input_length = iterator.get_next()
    
    return DataPipeline(initializer=iterator.initializer, input_data=src_input_ids, input_length=input_length,
        input_data_placeholder=src_data_placeholder, batch_size_placeholder=batch_size_placeholder)

def load_vocab_table(vocab_file,
                     vocab_size,
                     vocab_lookup,
                     unk,
                     sos,
                     eos,
                     pad):
    """load vocab table from vocab file"""
    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = 1
            
            if unk in vocab:
                del vocab[unk]
            if sos in vocab:
                del vocab[sos]
            if eos in vocab:
                del vocab[eos]
            if pad in vocab:
                del vocab[pad]
            
            if vocab_lookup is not None:
                vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
            
            sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
            sorted_vocab = [unk, sos, eos, pad] + sorted_vocab
            
            vocab_table = sorted_vocab[:vocab_size]
            vocab_size = len(vocab_table)
            
            vocab_index = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=0)
            vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=unk)
            
            return vocab_table, vocab_size, vocab_index, vocab_inverted_index
    else:
        raise FileNotFoundError("vocab file not found")

def create_vocab_table(text_file,
                       vocab_size,
                       vocab_lookup,
                       unk,
                       sos,
                       eos,
                       pad):
    """create vocab table from text file"""
    if tf.gfile.Exists(text_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(text_file, "rb")) as file:
            vocab = {}
            for line in file:
                words = line.strip().split(' ')
                for word in words:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
            
            if unk in vocab:
                del vocab[unk]
            if sos in vocab:
                del vocab[sos]
            if eos in vocab:
                del vocab[eos]
            if pad in vocab:
                del vocab[pad]
            
            if vocab_lookup is not None:
                vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
            
            sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
            sorted_vocab = [unk, sos, eos, pad] + sorted_vocab
            
            vocab_table = sorted_vocab[:vocab_size]
            vocab_size = len(vocab_table)
            
            vocab_index = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=0)
            vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=tf.constant(vocab_table), default_value=unk)
            
            return vocab_table, vocab_size, vocab_index, vocab_inverted_index
    else:
        raise FileNotFoundError("text file not found")

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not tf.gfile.Exists(vocab_dir):
        tf.gfile.MakeDirs(vocab_dir)
    
    if not tf.gfile.Exists(vocab_file):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "w")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not tf.gfile.Exists(embedding_dir):
        tf.gfile.MakeDirs(embedding_dir)
    
    if not tf.gfile.Exists(embedding_file):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(embedding_file, "w")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_input(text_file):
    """load data from text file"""
    input_table = []
    if tf.gfile.Exists(text_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(text_file, "rb")) as file:
            for line in file:
                input_table.append(line.strip())
            input_size = len(input_table)
            
            return input_table, input_size
    else:
        raise FileNotFoundError("text file not found")
