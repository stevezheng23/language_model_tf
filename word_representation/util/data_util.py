import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["LanguageModelPipeline", "create_lm_infer_pipeline", "create_lm_pipeline",
           "load_pretrained_embedding", "create_embedding_file", "convert_embedding",
           "load_vocab_table", "create_vocab_table", "create_vocab_file",
           "load_input", "prepare_data"]

class LanguageModelPipeline(collections.namedtuple("LanguageModelPipeline",
    ("initializer", "text_input", "text_output", "text_input_length", "text_output_length",
     "text_data_placeholder", "batch_size_placeholder"))):
    pass

def create_lm_infer_pipeline(vocab_index,
                             max_length,
                             sos,
                             eos,
                             pad):
    """create language model infer data pipeline based on config"""
    sos_id = tf.cast(vocab_index.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_index.lookup(tf.constant(eos)), tf.int32)
    pad_id = tf.cast(vocab_index.lookup(tf.constant(pad)), tf.int32)
    
    input_data_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
    
    dataset = tf.data.Dataset.from_tensor_slices(input_data_placeholder)
    dataset = dataset.map(lambda text: tf.string_split([text], delimiter=' ').values)
    dataset = dataset.filter(lambda text: tf.size(text) > 0)
    dataset = dataset.map(lambda text: text[:max_length])
    
    dataset = dataset.map(lambda text: tf.cast(vocab_index.lookup(text), tf.int32))
    dataset = dataset.map(lambda text: tf.concat((text, [eos_id]), 0))
    dataset = dataset.map(lambda text: (text, tf.size(text)))
    
    dataset = dataset.padded_batch(
        batch_size=batch_size_placeholder,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([])),
        padding_values=(
            pad_id,
            0))
    
    iterator = dataset.make_initializable_iterator()
    input_id, input_len = iterator.get_next()
    
    return LanguageModelPipeline(initializer=iterator.initializer,
        text_input=input_id, text_output=None, text_input_length=input_len,
        text_output_length=None, text_data_placeholder=input_data_placeholder,
        batch_size_placeholder=batch_size_placeholder)

def create_lm_pipeline(text_file,
                       vocab_index,
                       max_length,
                       sos,
                       eos,
                       pad,
                       batch_size,
                       random_seed,
                       enable_shuffle):
    """create language model data pipeline based on config"""
    sos_id = tf.cast(vocab_index.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_index.lookup(tf.constant(eos)), tf.int32)
    pad_id = tf.cast(vocab_index.lookup(tf.constant(pad)), tf.int32)
    
    dataset = tf.data.TextLineDataset([text_file])
    
    if enable_shuffle == True:
        buffer_size = batch_size * 1000
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.map(lambda text: tf.string_split([text], delimiter=' ').values)
    dataset = dataset.filter(lambda text: tf.size(text) > 0)
    dataset = dataset.map(lambda text: text[:max_length])
    
    dataset = dataset.map(lambda text: tf.cast(vocab_index.lookup(text), tf.int32))
    dataset = dataset.map(lambda text: (tf.concat((text, [eos_id]), 0), tf.concat(([sos_id], text), 0)))
    dataset = dataset.map(lambda text_input, text_output:
        (text_input, text_output, tf.size(text_input), tf.size(text_output)))
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([])),
        padding_values=(
            pad_id,
            pad_id,
            0,
            0))
    
    iterator = dataset.make_initializable_iterator()
    input_id, output_id, input_len, output_len = iterator.get_next()
    
    return LanguageModelPipeline(initializer=iterator.initializer,
        text_input=input_id, text_output=output_id, text_input_length=input_len,
        text_output_length=output_len, text_data_placeholder=None, batch_size_placeholder=None)

def load_pretrained_embedding(embedding_file,
                              embedding_size,
                              unk,
                              sos,
                              eos,
                              pad):
    """load pre-trained embeddings from embedding file"""
    if tf.gfile.Exists(embedding_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(embedding_file, "rb")) as file:
            embedding = {}
            for line in file:
                items = line.strip().split(' ')
                if len(items) != embedding_size + 1:
                    continue
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if word not in embedding:
                    embedding[word] = vector
            
            if unk not in embedding:
                embedding[unk] = np.random.rand(embedding_size)
            if sos not in embedding:
                embedding[sos] = np.random.rand(embedding_size)
            if eos not in embedding:
                embedding[eos] = np.random.rand(embedding_size)
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

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

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

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

def prepare_data(logger,
                 input_file,
                 vocab_file,
                 embedding_file,
                 full_embedding_file,
                 vocab_size,
                 embed_dim,
                 unk,
                 sos,
                 eos,
                 pad,
                 pretrained_embedding):
    """prepare input data from input files"""
    input_data = None
    if tf.gfile.Exists(input_file):
        logger.log_print("# loading input data from {0}".format(input_file))
        input_data, input_size = load_input(input_file)
        logger.log_print("# input data has {0} lines".format(input_size))
    
    embedding_data = None
    if pretrained_embedding == True:
        if tf.gfile.Exists(embedding_file):
            logger.log_print("# loading embeddings from {0}".format(embedding_file))
            embedding_data = load_pretrained_embedding(embedding_file, embed_dim, unk, sos, eos, pad)
        elif tf.gfile.Exists(full_embedding_file):
            logger.log_print("# loading embeddings from {0}".format(full_embedding_file))
            embedding_data = load_pretrained_embedding(full_embedding_file, embed_dim, unk, sos, eos, pad)
        
        embedding_size = len(embedding_data) if embedding_data is not None else 0
        logger.log_print("# embeddings has {0} words".format(embedding_size))
    
    if tf.gfile.Exists(vocab_file):
        logger.log_print("# loading vocabs from {0}".format(vocab_file))
        (vocab_table, vocab_size, vocab_index,
            vocab_inverted_index) = load_vocab_table(vocab_file,
            vocab_size, embedding_data, unk, sos, eos, pad)
    elif tf.gfile.Exists(input_file):
        logger.log_print("# creating vocabs from {0}".format(input_file))
        (vocab_table, vocab_size, vocab_index,
            vocab_inverted_index) = create_vocab_table(input_file,
            vocab_size, embedding_data, unk, sos, eos, pad)
        logger.log_print("# creating vocab file {0}".format(vocab_file))
        create_vocab_file(vocab_file, vocab_table)
    else:
        raise ValueError("{0} or {1} must be provided".format(vocab_file, input_file))
    logger.log_print("# vocab table has {0} words".format(vocab_size))
    
    if embedding_data is not None:
        embedding_data = { k: embedding_data[k] for k in vocab_table if k in embedding_data }
        logger.log_print("# embeddings has {0} words after filtering".format(len(embedding_data)))
        if not tf.gfile.Exists(embedding_file):
            logger.log_print("# creating embedding file {0}".format(embedding_file))
            create_embedding_file(embedding_file, embedding_data)
        embedding_data = convert_embedding(embedding_data)
    
    return input_data, embedding_data, vocab_size, vocab_index, vocab_inverted_index
