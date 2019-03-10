import argparse
import os
import os.path

from util.data_util import *

def add_arguments(parser):
    parser.add_argument("--input_dir", help="input directory", required=True)
    parser.add_argument("--max_word_size", help="maximum word size", type=int, required=False, default=512)
    parser.add_argument("--max_char_size", help="maximum char size", type=int, required=False, default=16)
    parser.add_argument("--full_embedding_file", help="full embedding file", required=True)
    parser.add_argument("--word_embedding_file", help="word embedding file", required=True)
    parser.add_argument("--word_embed_dim", help="word embedding dimension", type=int, required=False, default=300)
    parser.add_argument("--word_vocab_file", help="word voacb file", required=True)
    parser.add_argument("--word_vocab_size", help="word vocab size", type=int, required=False, default=100000)
    parser.add_argument("--word_unk", help="unknown word", required=False, default="<unk>")
    parser.add_argument("--word_pad", help="padding word", required=False, default="<pad>")
    parser.add_argument("--word_sos", help="word sos token", required=False, default="<s>")
    parser.add_argument("--word_eos", help="word eos token", required=False, default="</s>")
    parser.add_argument("--char_vocab_file", help="char vocab file", required=True)
    parser.add_argument("--char_vocab_size", help="char vocab size", type=int, required=False, default=1000)
    parser.add_argument("--char_unk", help="unknown char", required=False, default="*")
    parser.add_argument("--char_pad", help="padding char", required=False, default="#")

def prepare_resource(input_dir,
                     max_word_size,
                     max_char_size,
                     full_embedding_file,
                     word_embedding_file,
                     word_embed_dim,
                     word_vocab_file,
                     word_vocab_size,
                     word_unk,
                     word_pad,
                     word_sos,
                     word_eos,
                     char_vocab_file,
                     char_vocab_size,
                     char_unk,
                     char_pad):
    print("# loading embeddings from {0}".format(full_embedding_file))
    word_embedding_data = load_embedding_file(full_embedding_file, word_embed_dim, word_unk, word_pad, word_sos, word_eos)
    print("# word embedding table has {0} words".format(len(word_embedding_data)))
    
    if not os.path.isdir(input_dir):
        raise FileNotFoundError("input directory not found")
    
    word_vocab_lookup = {}
    char_vocab_lookup = {}
    input_files = os.listdir(input_dir)
    for input_file in input_files:
        input_file = os.path.join(input_dir, input_file)
        if not os.path.isfile(input_file):
            continue
        
        print("# generating vocab from {0}".format(input_file))
        input_data = load_data(input_file)
        word_vocab = create_word_vocab(input_data)
        char_vocab = create_char_vocab(input_data)
        
        for word in word_vocab.keys():
            if word in word_vocab_lookup:
                word_vocab_lookup[word] += word_vocab[word]
            else:
                word_vocab_lookup[word] = word_vocab[word]
        
        for char in char_vocab.keys():
            if char in char_vocab_lookup:
                char_vocab_lookup[char] += char_vocab[char]
            else:
                char_vocab_lookup[char] = char_vocab[char]
    
    print("# processing word vocab table")
    (word_vocab_table, word_vocab_size, word_vocab_index,
        word_vocab_inverted_index) = process_vocab_table(word_vocab_lookup,
        word_vocab_size, 0, word_embedding_data, word_unk, word_pad, word_sos, word_eos)
    print("# processing char vocab table")
    (char_vocab_table, char_vocab_size, char_vocab_index,
        char_vocab_inverted_index) = process_vocab_table(char_vocab_lookup,
        char_vocab_size, 0, None, char_unk, char_pad)
    
    print("# creating word vocab file {0}".format(word_vocab_file))
    create_vocab_file(word_vocab_file, word_vocab_table)
    print("# creating char vocab file {0}".format(char_vocab_file))
    create_vocab_file(char_vocab_file, char_vocab_table)
    
    word_embedding_data = { k: word_embedding_data[k] for k in word_vocab_table if k in word_embedding_data }
    print("# word embedding table has {0} words after filtering".format(len(word_embedding_data)))
    
    print("# creating word embedding file {0}".format(word_embedding_file))
    create_embedding_file(word_embedding_file, word_embedding_data)

def main(args):
    prepare_resource(args.input_dir, args.max_word_size, args.max_char_size,
        args.full_embedding_file, args.word_embedding_file, args.word_embed_dim, args.word_vocab_file,
        args.word_vocab_size, args.word_unk, args.word_pad, args.word_sos, args.word_eos,
        args.char_vocab_file, args.char_vocab_size, args.char_unk, args.char_pad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
