import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["DataPipeline", "create_dynamic_pipeline", "create_data_pipeline",
           "create_text_dataset", "get_text_dataset", "generate_word_feat", "generate_char_feat",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_file", "load_vocab_file", "process_vocab_table",
           "create_word_vocab", "create_char_vocab", "load_data", "prepare_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "word_vocab_size", "char_vocab_size", "input_text_word", "input_text_char",
     "input_text_word_mask", "input_text_char_mask", "word_vocab_inverted_index",
     "input_text_placeholder", "data_size_placeholder", "batch_size_placeholder"))):
    pass

def create_dynamic_pipeline(input_text_word_dataset,
                            input_text_char_dataset,
                            word_vocab_size,
                            word_vocab_index,
                            word_vocab_inverted_index,
                            word_pad,
                            word_feat_enable,
                            char_vocab_size,
                            char_vocab_index,
                            char_pad,
                            char_feat_enable,
                            input_text_placeholder,
                            data_size_placeholder,
                            batch_size_placeholder):
    """create dynamic data pipeline"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = default_pad_id
        input_text_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = default_pad_id
        input_text_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        
    dataset = tf.data.Dataset.zip((input_text_word_dataset, input_text_char_dataset))
    
    dataset = dataset.batch(batch_size=batch_size_placeholder)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_text_word = batch_data[0]
        input_text_word_mask = tf.cast(tf.not_equal(batch_data[0], word_pad_id), dtype=tf.float32)
    else:
        input_text_word = None
        input_text_word_mask = None
    
    if char_feat_enable == True:
        input_text_char = batch_data[1]
        input_text_char_mask = tf.cast(tf.not_equal(batch_data[1], char_pad_id), dtype=tf.float32)
    else:
        input_text_char = None
        input_text_char_mask = None
    
    return DataPipeline(initializer=iterator.initializer,
        word_vocab_size=word_vocab_size, char_vocab_size=char_vocab_size,
        input_text_word=input_text_word, input_text_char=input_text_char,
        input_text_word_mask=input_text_word_mask, input_text_char_mask=input_text_char_mask,
        word_vocab_inverted_index=word_vocab_inverted_index, input_text_placeholder=input_text_placeholder,
        data_size_placeholder=data_size_placeholder, batch_size_placeholder=batch_size_placeholder)

def create_data_pipeline(input_text_word_dataset,
                         input_text_char_dataset,
                         word_vocab_size,
                         word_vocab_index,
                         word_vocab_inverted_index,
                         word_pad,
                         word_feat_enable,
                         char_vocab_size,
                         char_vocab_index,
                         char_pad,
                         char_feat_enable,
                         enable_shuffle,
                         buffer_size,
                         data_size,
                         batch_size,
                         random_seed):
    """create data pipeline"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = default_pad_id
        input_text_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = default_pad_id
        input_text_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    dataset = tf.data.Dataset.zip((input_text_word_dataset, input_text_char_dataset))
    
    if enable_shuffle == True:
        buffer_size = min(buffer_size, data_size)
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_text_word = batch_data[0]
        input_text_word_mask = tf.cast(tf.not_equal(batch_data[0], word_pad_id), dtype=tf.float32)
    else:
        input_text_word = None
        input_text_word_mask = None
    
    if char_feat_enable == True:
        input_text_char = batch_data[1]
        input_text_char_mask = tf.cast(tf.not_equal(batch_data[1], char_pad_id), dtype=tf.float32)
    else:
        input_text_char = None
        input_text_char_mask = None
    
    return DataPipeline(initializer=iterator.initializer,
        word_vocab_size=word_vocab_size, char_vocab_size=char_vocab_size,
        input_text_word=input_text_word, input_text_char=input_text_char,
        input_text_word_mask=input_text_word_mask, input_text_char_mask=input_text_char_mask,
        word_vocab_inverted_index=word_vocab_inverted_index, input_text_placeholder=None,
        data_size_placeholder=None, batch_size_placeholder=None)

def create_text_dataset(input_data_set,
                        word_vocab_index,
                        word_max_size,
                        word_pad,
                        word_sos,
                        word_eos,
                        word_feat_enable,
                        char_vocab_index,
                        char_max_size,
                        char_pad,
                        char_feat_enable):
    """create word/char-level dataset for input data"""
    word_dataset = None
    if word_feat_enable == True:
        word_dataset = input_data_set.map(lambda sent: generate_word_feat(sent,
            word_vocab_index, word_max_size, word_pad, word_sos, word_eos))
    
    char_dataset = None
    if char_feat_enable == True:
        char_dataset = input_data_set.map(lambda sent: generate_char_feat(sent,
            word_max_size, word_sos, word_eos, char_vocab_index, char_max_size, char_pad))
    
    return word_dataset, char_dataset

def get_text_dataset(input_path):
    if os.path.isdir(input_path):
        input_files = os.listdir(input_path)
        input_files = [os.path.join(input_path, input_file) for input_file in input_files]
    else:
        if os.path.exists(input_path):
            input_files = [input_path]
        else:
            raise FileNotFoundError("input file not found")
    
    text_dataset = tf.data.TextLineDataset(input_files)
    
    return text_dataset

def generate_word_feat(sentence,
                       word_vocab_index,
                       word_max_size,
                       word_pad,
                       word_sos,
                       word_eos):
    """process words for sentence"""
    sentence_words = tf.string_split([sentence], delimiter=' ').values
    word_prefix = tf.convert_to_tensor([word_sos])
    word_suffix = tf.convert_to_tensor([word_eos] + [word_pad] * word_max_size)
    sentence_words = tf.concat([word_prefix, sentence_words[:word_max_size], word_suffix], axis=0)
    sentence_words = tf.reshape(sentence_words[:word_max_size + 2], shape=[word_max_size + 2])
    sentence_words = tf.cast(word_vocab_index.lookup(sentence_words), dtype=tf.int32)
    sentence_words = tf.expand_dims(sentence_words, axis=-1)
    
    return sentence_words

def generate_char_feat(sentence,
                       word_max_size,
                       word_sos,
                       word_eos,
                       char_vocab_index,
                       char_max_size,
                       char_pad):
    """generate characters for sentence"""
    def word_to_char(word):
        """process characters for word"""
        word_chars = tf.string_split([word], delimiter='').values
        word_chars = tf.concat([word_chars[:char_max_size],
            tf.constant(char_pad, shape=[char_max_size])], axis=0)
        word_chars = tf.reshape(word_chars[:char_max_size], shape=[char_max_size])
        
        return word_chars
    
    sentence_words = tf.string_split([sentence], delimiter=' ').values
    word_prefix = tf.convert_to_tensor([word_sos])
    word_suffix = tf.convert_to_tensor([word_eos] + [char_pad] * word_max_size)
    sentence_words = tf.concat([word_prefix, sentence_words[:word_max_size], word_suffix], axis=0)
    sentence_words = tf.reshape(sentence_words[:word_max_size + 2], shape=[word_max_size + 2])
    sentence_chars = tf.map_fn(word_to_char, sentence_words)
    sentence_chars = tf.cast(char_vocab_index.lookup(sentence_chars), dtype=tf.int32)
    
    return sentence_chars

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    
    if not os.path.exists(embedding_file):
        with codecs.getwriter("utf-8")(open(embedding_file, "wb")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        pad,
                        sos=None,
                        eos=None):
    """load pre-train embeddings from embedding file"""
    if os.path.exists(embedding_file):
        with codecs.getreader("utf-8")(open(embedding_file, "rb")) as file:
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
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            if sos and sos not in embedding:
                embedding[sos] = np.random.rand(embedding_size)
            if eos and eos not in embedding:
                embedding[eos] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    
    if not os.path.exists(vocab_file):
        with codecs.getwriter("utf-8")(open(vocab_file, "wb")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
    if os.path.exists(vocab_file):
        with codecs.getreader("utf-8")(open(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = MAX_INT
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_vocab_table(vocab,
                        vocab_size,
                        vocab_threshold,
                        vocab_lookup,
                        unk,
                        pad,
                        sos=None,
                        eos=None):
    """process vocab table"""
    if unk == pad:
        raise ValueError("unknown vocab {0} can not be the same as padding vocab {1}".format(unk, pad))
    
    default_vocab = [unk, pad]
    if sos:
        default_vocab.append(sos)
    if eos:
        default_vocab.append(eos)
    
    if unk in vocab:
        del vocab[unk]
    if pad in vocab:
        del vocab[pad]
    if sos and sos in vocab:
        del vocab[sos]
    if eos and eos in vocab:
        del vocab[eos]
    
    vocab = { k: vocab[k] for k in vocab.keys() if vocab[k] >= vocab_threshold }
    if vocab_lookup is not None:
        vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
    
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    vocab_table = default_vocab + sorted_vocab[:vocab_size-len(default_vocab)]
    vocab_size = len(vocab_table)
    
    vocab_index = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=0)
    vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=unk)
    
    return vocab_table, vocab_size, vocab_index, vocab_inverted_index

def create_word_vocab(input_data):
    """create word vocab from input data"""
    word_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            if word not in word_vocab:
                word_vocab[word] = 1
            else:
                word_vocab[word] += 1
    
    return word_vocab

def create_char_vocab(input_data):
    """create char vocab from input data"""
    char_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            chars = list(word)
            for ch in chars:
                if ch not in char_vocab:
                    char_vocab[ch] = 1
                else:
                    char_vocab[ch] += 1
    
    return char_vocab

def load_data(input_path):
    """load data from file/folder"""
    if os.path.isdir(input_path):
        input_files = os.listdir(input_path)
        input_files = [os.path.join(input_path, input_file) for input_file in input_files]
    else:
        if os.path.exists(input_path):
            input_files = [input_path]
        else:
            raise FileNotFoundError("input file not found")
    
    text_data = []
    for input_file in input_files:
        with codecs.getreader("utf-8")(open(input_file, "rb")) as file:
            for line in file:
                text = line.strip()
                text_data.append(text)
    
    return text_data

def prepare_data(logger,
                 input_path,
                 word_vocab_file,
                 word_vocab_size,
                 word_vocab_threshold,
                 word_embed_dim,
                 word_embed_file,
                 full_word_embed_file,
                 word_unk,
                 word_pad,
                 word_sos,
                 word_eos,
                 word_feat_enable,
                 pretrain_word_embed,
                 char_vocab_file,
                 char_vocab_size,
                 char_vocab_threshold,
                 char_unk,
                 char_pad,
                 char_feat_enable):
    """prepare data"""
    logger.log_print("# loading input data from {0}".format(input_path))
    input_data = load_data(input_path)
    input_data_size = len(input_data)
    logger.log_print("# input data has {0} lines".format(input_data_size))
    
    word_embed_data = None
    if pretrain_word_embed == True:
        if os.path.exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file, word_embed_dim, word_unk, word_pad, word_sos, word_eos)
        elif os.path.exists(full_word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(full_word_embed_file))
            word_embed_data = load_embedding_file(full_word_embed_file, word_embed_dim, word_unk, word_pad, word_sos, word_eos)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, full_word_embed_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    word_vocab = None
    word_vocab_index = None
    word_vocab_inverted_index = None
    if os.path.exists(word_vocab_file):
        logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
        word_vocab = load_vocab_file(word_vocab_file)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad, word_sos, word_eos)
    elif input_data is not None:
        logger.log_print("# creating word vocab table from input data")
        word_vocab = create_word_vocab(input_data)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad, word_sos, word_eos)
        logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
        create_vocab_file(word_vocab_file, word_vocab_table)
    else:
        raise ValueError("{0} or input data must be provided".format(word_vocab_file))
    
    logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    char_vocab = None
    char_vocab_index = None
    char_vocab_inverted_index = None
    if char_feat_enable is True:
        if os.path.exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (_, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
        elif input_data is not None:
            logger.log_print("# creating char vocab table from input data")
            char_vocab = create_char_vocab(input_data)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or input data must be provided".format(char_vocab_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not os.path.exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (input_data, word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)
