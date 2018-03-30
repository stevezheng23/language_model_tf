import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["DataPipeline", "create_data_pipeline",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_table", "create_vocab_file", "load_vocab_file",
           "process_word", "process_subword", "process_char",
           "process_input_data", "load_input_data", "prepare_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "input_word_feat", "input_subword_feat", "input_char_feat",
     "input_word_mask", "input_subword_mask", "input_char_mask",
     "word_feat_placeholder", "subword_feat_placeholder", "char_feat_placeholder"))):
    pass

def create_data_pipeline(word_vocab_index,
                         subword_vocab_index,
                         char_vocab_index,
                         batch_size,
                         random_seed):
    """create data pipeline for word/subword/char-level representation"""    
    word_feat_placeholder = tf.placeholder(shape=[None, None], dtype=tf.string)
    subword_feat_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.string)
    char_feat_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.string)

    word_feat_dataset = tf.data.Dataset.from_tensor_slices(word_feat_placeholder)
    subword_feat_dataset = tf.data.Dataset.from_tensor_slices(subword_feat_placeholder)
    char_feat_dataset = tf.data.Dataset.from_tensor_slices(char_feat_placeholder)
    dataset = tf.data.Dataset.zip((word_feat_dataset, subword_feat_dataset, char_feat_dataset))
    
    #buffer_size = batch_size * 1000
    #dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.map(lambda word_feat, subword_feat, char_feat: (tf.cast(word_vocab_index.lookup(word_feat), tf.int32),
         tf.cast(subword_vocab_index.lookup(subword_feat), tf.int32), tf.cast(char_vocab_index.lookup(char_feat), tf.int32)))
    
    dataset = dataset.batch(batch_size=batch_size)
    
    iterator = dataset.make_initializable_iterator()
    input_word_feat, input_subword_feat, input_char_feat = iterator.get_next()
    
    return DataPipeline(initializer=iterator.initializer, input_word_feat=input_word_feat,
        input_subword_feat=input_subword_feat, input_char_feat=input_char_feat,
        input_word_mask=None, input_subword_mask=None, input_char_mask=None,
        word_feat_placeholder=word_feat_placeholder, subword_feat_placeholder=subword_feat_placeholder,
        char_feat_placeholder=char_feat_placeholder)

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

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        sos,
                        eos,
                        pad):
    """load pre-train embeddings from embedding file"""
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

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

def create_vocab_table(vocab,
                       vocab_size,
                       vocab_lookup,
                       unk,
                       sos,
                       eos,
                       pad):
    """create vocab table from vocab data"""
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

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
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
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_word(words,
                 word_max_length,
                 word_sos,
                 word_eos,
                 word_pad,
                 word_vocab):
    """process words for sentence"""
    for word in words:
        if word not in word_vocab:
            word_vocab[word] = 1
        else:
            word_vocab[word] += 1
    
    word_feat = np.full((word_max_length+2), word_pad)
    words = [word_sos] + words[:word_max_length] + [word_eos]
    word_length = len(words)
    word_feat[:word_length] = words
    return word_feat

def process_subword(words,
                    word_max_length,
                    subword_max_length,
                    word_sos,
                    word_eos,
                    subword_sos,
                    subword_eos,
                    subword_pad,
                    subword_vocab):
    """process subwords for sentence"""   
    for word in words:
        subwords = list(word)
        for subword in subwords:
            if subword not in subword_vocab:
                subword_vocab[subword] = 1
            else:
                subword_vocab[subword] += 1
    
    subword_feat = np.full((word_max_length+2, subword_max_length+2), subword_pad)
    words = [word_sos] + words[:word_max_length] + [word_eos]
    for i, word in enumerate(words):
        subwords = list(word)
        subwords = [subword_sos] + subwords[:subword_max_length] + [subword_eos]
        subword_length = len(subwords)
        subword_feat[i,:subword_length] = subwords
    
    return subword_feat

def process_char(words,
                 word_max_length,
                 char_max_length,
                 word_sos,
                 word_eos,
                 char_sos,
                 char_eos,
                 char_pad,
                 char_vocab):
    """process characters for sentence"""   
    for word in words:
        chars = list(word)
        for ch in chars:
            if ch not in char_vocab:
                char_vocab[ch] = 1
            else:
                char_vocab[ch] += 1
    
    char_feat = np.full((word_max_length+2, char_max_length+2), char_pad)
    words = [word_sos] + words[:word_max_length] + [word_eos]
    for i, word in enumerate(words):
        chars = list(word)
        chars = [char_sos] + chars[:char_max_length] + [char_eos]
        char_length = len(chars)
        char_feat[i,:char_length] = chars
    
    return char_feat

def process_input_data(input_data,
                       input_size,
                       word_max_length,
                       word_feat_enable,
                       word_sos,
                       word_eos,
                       word_pad,
                       subword_max_length,
                       subword_feat_enable,
                       subword_sos,
                       subword_eos,
                       subword_pad,
                       char_max_length,
                       char_feat_enable,
                       char_sos,
                       char_eos,
                       char_pad):
    """process input data for featurization"""
    word_feat_data = np.full((input_size, word_max_length+2), word_pad)
    subword_feat_data = np.full((input_size, word_max_length+2, subword_max_length+2), subword_pad)
    char_feat_data = np.full((input_size, word_max_length+2, char_max_length+2), char_pad)
    word_vocab = {}
    subword_vocab = {}
    char_vocab = {}
    for i, sentence in enumerate(input_data):
        words = sentence.strip().split(' ')
        if word_feat_enable is True:
            word_feat = process_word(words, word_max_length,
                word_sos, word_eos, word_pad, word_vocab)
            word_feat_data[i,:] = word_feat
        if subword_feat_enable is True:
            subword_feat = process_subword(words, word_max_length, subword_max_length,
                word_sos, word_eos, subword_sos, subword_eos, subword_pad, subword_vocab)
            subword_feat_data[i,:,:] = subword_feat
        if char_feat_enable is True:
            char_feat = process_char(words, word_max_length, char_max_length,
                word_sos, word_eos, char_sos, char_eos, char_pad, char_vocab)
            char_feat_data[i,:,:] = char_feat
    
    return word_feat_data, subword_feat_data, char_feat_data, word_vocab, subword_vocab, char_vocab

def load_input_data(input_file):
    """load input data from input file"""
    input_data = []
    if tf.gfile.Exists(input_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(input_file, "rb")) as file:
            for line in file:
                input_data.append(line.strip())
            input_size = len(input_data)
            
            return input_data, input_size
    else:
        raise FileNotFoundError("input file not found")

def prepare_data(logger,
                 input_file,
                 word_vocab_file,
                 word_vocab_size,
                 word_embed_dim,
                 word_embed_file,
                 full_word_embed_file,
                 word_unk,
                 word_sos,
                 word_eos,
                 word_pad,
                 word_max_length,
                 word_feat_enable,
                 pretrain_word_embed,
                 subword_vocab_file,
                 subword_vocab_size,
                 subword_embed_dim,
                 subword_unk,
                 subword_sos,
                 subword_eos,
                 subword_pad,
                 subword_max_length,
                 subword_feat_enable,
                 char_vocab_file,
                 char_vocab_size,
                 char_embed_dim,
                 char_unk,
                 char_sos,
                 char_eos,
                 char_pad,
                 char_max_length,
                 char_feat_enable):
    """prepare data for word representation"""
    input_data = None
    if tf.gfile.Exists(input_file):
        logger.log_print("# loading input data from {0}".format(input_file))
        input_data, input_size = load_input_data(input_file)
        logger.log_print("# input data has {0} lines".format(input_size))
    
    word_feat_data = None
    subword_feat_data = None
    char_feat_data = None
    word_vocab = None
    subword_vocab = None
    char_vocab = None
    if input_data is not None:
        (word_feat_data, subword_feat_data, char_feat_data, word_vocab,
            subword_vocab, char_vocab) = process_input_data(input_data, input_size, word_max_length, word_feat_enable,
            word_sos, word_eos, word_pad, subword_max_length, subword_feat_enable, subword_sos, subword_eos, subword_pad,
            char_max_length, char_feat_enable, char_sos, char_eos, char_pad)
    
    word_embed_data = None
    if pretrain_word_embed == True:
        if tf.gfile.Exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file,
                word_embed_dim, word_unk, word_sos, word_eos, word_pad)
        elif tf.gfile.Exists(full_word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(full_word_embed_file))
            word_embed_data = load_embedding_file(full_word_embed_file,
                word_embed_dim, word_unk, word_sos, word_eos, word_pad)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, full_word_embed_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    if word_feat_enable is True:
        if tf.gfile.Exists(word_vocab_file):
            logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
            word_vocab = load_vocab_file(word_vocab_file)
            (word_vocab_table, word_vocab_size, word_vocab_index,
                word_vocab_inverted_index) = create_vocab_table(word_vocab,
                word_vocab_size, word_embed_data, word_unk, word_sos, word_eos, word_pad)
        elif word_vocab is not None:
            logger.log_print("# creating word vocab table from {0}".format(input_file))
            (word_vocab_table, word_vocab_size, word_vocab_index,
                word_vocab_inverted_index) = create_vocab_table(word_vocab,
                word_vocab_size, word_embed_data, word_unk, word_sos, word_eos, word_pad)
            logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
            create_vocab_file(word_vocab_file, word_vocab_table)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, input_file))

        logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    if subword_feat_enable is True:
        if tf.gfile.Exists(subword_vocab_file):
            logger.log_print("# loading subword vocab table from {0}".format(subword_vocab_file))
            subword_vocab = load_vocab_file(subword_vocab_file)
            (subword_vocab_table, subword_vocab_size, subword_vocab_index,
                subword_vocab_inverted_index) = create_vocab_table(subword_vocab,
                subword_vocab_size, None, subword_unk, subword_sos, subword_eos, subword_pad)
        elif subword_vocab is not None:
            logger.log_print("# creating subword vocab table from {0}".format(input_file))
            (subword_vocab_table, subword_vocab_size, subword_vocab_index,
                subword_vocab_inverted_index) = create_vocab_table(subword_vocab,
                subword_vocab_size, None, subword_unk, subword_sos, subword_eos, subword_pad)
            logger.log_print("# creating subword vocab file {0}".format(subword_vocab_file))
            create_vocab_file(subword_vocab_file, subword_vocab_table)
        else:
            raise ValueError("{0} or {1} must be provided".format(subword_vocab_file, input_file))

        logger.log_print("# subword vocab table has {0} subwords".format(subword_vocab_size))
    
    if char_feat_enable is True:
        if tf.gfile.Exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = create_vocab_table(char_vocab,
                char_vocab_size, None, char_unk, char_sos, char_eos, char_pad)
        elif char_vocab is not None:
            logger.log_print("# creating char vocab table from {0}".format(input_file))
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = create_vocab_table(char_vocab,
                char_vocab_size, None, char_unk, char_sos, char_eos, char_pad)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or {1} must be provided".format(char_vocab_file, input_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not tf.gfile.Exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (word_feat_data, subword_feat_data, char_feat_data, word_embed_data,
        word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)
